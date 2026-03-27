"""
Reinforcement-learning environment for the Submodular Cover Sensor-Selection problem.

Problem statement
-----------------
Given:
    J     : N×N real symmetric positive-definite Toeplitz matrix (Bessel kernel)
    sigma : noise standard deviation
    epsilon : trace-constraint threshold

Find a minimum binary vector w ∈ {0,1}^N such that

    trace( J - J·W·(W·J·W + sigma²·I)⁻¹·W·J ) ≤ epsilon

where W = diag(w).

This module casts the problem as a sequential MDP:
  - State   : graph (J) with per-node features (current selection, posterior std, etc.)
  - Action  : select an unselected node  (set w_i ← 1)
  - Reward  : shaped reward encouraging constraint satisfaction with few sensors
  - Terminal: when trace ≤ epsilon OR all nodes are selected
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class SensorSelectionEnv:
    """Gym-style environment for the sensor-selection / submodular-cover problem.

    Parameters
    ----------
    J : np.ndarray
        N×N positive-definite Toeplitz covariance matrix.
    sigma : float
        Observation noise standard deviation.
    epsilon : float
        Maximum allowed posterior trace.
    """

    # ── type aliases ──
    NodeFeatures = np.ndarray   # shape (N, NODE_FEAT_DIM)
    State = Tuple[NodeFeatures, np.ndarray, float, np.ndarray]
    # (node_features, J, current_trace, selected_mask)

    def __init__(self, J: np.ndarray, sigma: float, epsilon: float) -> None:
        assert J.ndim == 2 and J.shape[0] == J.shape[1], "J must be square"
        self.J: np.ndarray = J.copy()
        self.N: int = J.shape[0]
        self.sigma: float = float(sigma)
        self.epsilon: float = float(epsilon)

        self.trace_J: float = float(np.trace(J))
        self.diag_J: np.ndarray = np.diag(J)          # prior variances

        # Running state (initialised by reset())
        self.selected: np.ndarray = np.zeros(self.N, dtype=bool)
        self.current_trace: float = self.trace_J
        self.step_count: int = 0

        # ── precompute a correlation-based adjacency matrix for GNN ──
        # We normalise by the diagonal so that adj[i,i] = 0 and |adj[i,j]| ≤ 1
        diag_sqrt = np.sqrt(self.diag_J)
        denom = diag_sqrt[:, None] * diag_sqrt[None, :]
        self._corr: np.ndarray = J / (denom + 1e-12)
        np.fill_diagonal(self._corr, 0.0)             # remove self-loops
        row_sum = self._corr.sum(axis=1, keepdims=True)
        # Normalised adjacency (row-stochastic, zero diagonal)
        self.adj_norm: np.ndarray = self._corr / (row_sum + 1e-12)

    # ──────────────────────────────────────────────────────────────────────────
    # Core environment API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> "SensorSelectionEnv.State":
        """Reset the environment to the initial state (no sensors selected)."""
        self.selected = np.zeros(self.N, dtype=bool)
        self.current_trace = self.trace_J
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> Tuple["SensorSelectionEnv.State", float, bool, dict]:
        """Select sensor at position *action* (set w[action] = 1).

        Parameters
        ----------
        action : int
            Index of the sensor to add (must be currently unselected).

        Returns
        -------
        state, reward, done, info
        """
        if self.selected[action]:
            raise ValueError(f"Node {action} is already selected.")

        prev_trace = self.current_trace
        self.selected[action] = True
        self.step_count += 1

        # Recompute posterior trace
        self.current_trace = self._compute_posterior_trace(
            np.where(self.selected)[0]
        )

        done = bool(self.current_trace <= self.epsilon)
        if self.step_count >= self.N and not done:
            done = True   # exhausted all nodes without satisfying the constraint

        # ── Reward shaping ────────────────────────────────────────────────────
        # +Δtrace/trace_J  : normalised trace reduction (greedy submodular gain)
        # -1/N             : small per-sensor cost (encourages parsimony)
        # +large bonus      : when constraint first satisfied
        delta_trace = prev_trace - self.current_trace          # ≥ 0
        gain_reward = delta_trace / (self.trace_J + 1e-12)     # in [0, 1]
        cost_reward = -1.0 / self.N

        reward = gain_reward + cost_reward
        if done and self.current_trace <= self.epsilon:
            # Additional bonus inversely proportional to number of sensors used
            reward += 1.0 + (self.N - self.step_count) / self.N

        info = {
            "trace": self.current_trace,
            "n_selected": self.step_count,
            "satisfied": self.current_trace <= self.epsilon,
            "trace_reduction": delta_trace,
        }
        return self._get_state(), reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_posterior_trace(self, selected_indices: np.ndarray) -> float:
        """Compute trace( J − J·W·(W·J·W + σ²·I)⁻¹·W·J ) for the given selection.

        Uses the matrix-inversion identity:
            trace_posterior = trace(J) − trace( (J_SS + σ²·I)⁻¹ · J_S^T · J_S )
        where J_S = J[:, S]  and  J_SS = J[S, :][:, S].
        """
        if len(selected_indices) == 0:
            return self.trace_J

        S = selected_indices
        J_S = self.J[:, S]                             # N × |S|
        J_SS = self.J[np.ix_(S, S)]                   # |S| × |S|
        M = J_SS + (self.sigma ** 2) * np.eye(len(S)) # |S| × |S|

        # Stable solve rather than explicit inversion
        try:
            reduction = np.trace(J_S.T @ np.linalg.solve(M, J_S.T).T)
        except np.linalg.LinAlgError:
            reduction = np.trace(J_S @ np.linalg.lstsq(M, J_S.T, rcond=None)[0])

        return float(self.trace_J - reduction)

    def _get_state(self) -> "SensorSelectionEnv.State":
        """Compute per-node features and return the full state tuple.

        Node feature vector (4 dimensions)
        -----------------------------------
        0 : is_selected  — binary (0/1)
        1 : diag_J[i] / diag_J[0]  — normalised prior variance
        2 : correlation_with_selected  — sum |corr(i, j)| for j ∈ S, normalised
        3 : current_trace / trace_J   — global coverage progress (same for all nodes)
        """
        node_features = np.zeros((self.N, 4), dtype=np.float32)

        node_features[:, 0] = self.selected.astype(np.float32)
        node_features[:, 1] = (self.diag_J / (self.diag_J[0] + 1e-12)).astype(np.float32)

        # Feature 2: sum of absolute correlations with currently selected set
        selected_indices = np.where(self.selected)[0]
        if len(selected_indices) > 0:
            corr_with_selected = np.abs(self._corr[:, selected_indices]).sum(axis=1)
            corr_with_selected /= (len(selected_indices) + 1e-12)
            node_features[:, 2] = corr_with_selected.astype(np.float32)

        # Feature 3: global trace progress (broadcast scalar)
        node_features[:, 3] = float(self.current_trace / (self.trace_J + 1e-12))

        return node_features, self.adj_norm.astype(np.float32), self.current_trace, self.selected.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())

    @property
    def is_satisfied(self) -> bool:
        return self.current_trace <= self.epsilon

    def action_mask(self) -> np.ndarray:
        """Boolean mask of shape (N,): True where action is valid (node not yet selected)."""
        return ~self.selected
