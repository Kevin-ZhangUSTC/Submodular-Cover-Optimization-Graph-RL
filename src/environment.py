"""
Reinforcement-learning environment for the Submodular Cover Sensor-Selection problem.

Problem statement
-----------------
Given:
    J     : N×N real symmetric Toeplitz matrix (Bessel / J0 kernel)
    sigma : noise standard deviation
    epsilon : trace-constraint threshold

Find a minimum binary vector w in {0,1}^N such that

    trace( J - J*W*(W*J*W + sigma^2*I)^{-1}*W*J ) <= epsilon

where W = diag(w).

This module casts the problem as a sequential MDP:
  - State   : graph (J) with per-node features (current selection, posterior std, etc.)
  - Action  : select an unselected node  (set w_i = 1)
  - Reward  : shaped reward encouraging constraint satisfaction with few sensors
  - Terminal: when trace <= epsilon OR all nodes are selected

When the kernel matrix is not positive-definite (e.g. J0 kernel), a small
diagonal regularization is applied internally to ensure numerically stable
linear solves, and the ``j_is_pd`` attribute is set to ``False`` to inform
downstream code.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import List, Tuple

from .kernel import is_positive_definite, regularize_matrix


class SensorSelectionEnv:
    """Gym-style environment for the sensor-selection / submodular-cover problem.

    Parameters
    ----------
    J : np.ndarray
        N x N Toeplitz covariance matrix.  Need not be positive-definite; if
        it is not, a regularised copy is used internally for posterior-trace
        computation and the ``j_is_pd`` flag is set to ``False``.
    sigma : float
        Observation noise standard deviation.
    epsilon : float
        Maximum allowed posterior trace.
    reg_min_eig : float
        Minimum eigenvalue used when regularising a non-PD kernel matrix.
    """

    # -- type aliases --
    NodeFeatures = np.ndarray   # shape (N, NODE_FEAT_DIM)
    State = Tuple[NodeFeatures, np.ndarray, float, np.ndarray]
    # (node_features, adj_norm, current_trace, selected_mask)

    def __init__(
        self,
        J: np.ndarray,
        sigma: float,
        epsilon: float,
        reg_min_eig: float = 1e-6,
    ) -> None:
        assert J.ndim == 2 and J.shape[0] == J.shape[1], "J must be square"
        self.J: np.ndarray = J.copy()
        self.N: int = J.shape[0]
        self.sigma: float = float(sigma)
        self.epsilon: float = float(epsilon)

        self.trace_J: float = float(np.trace(J))
        self.diag_J: np.ndarray = np.diag(J)          # prior variances

        # Check PD and, if needed, create a regularised version for solves
        self.j_is_pd: bool = is_positive_definite(J)
        if not self.j_is_pd:
            warnings.warn(
                "Kernel matrix J is not positive-definite. "
                "A regularised copy will be used for posterior-trace computation.",
                UserWarning,
                stacklevel=2,
            )
            self._J_reg: np.ndarray = regularize_matrix(J, min_eig=reg_min_eig)
        else:
            self._J_reg = self.J

        # Running state (initialised by reset())
        self.selected: np.ndarray = np.zeros(self.N, dtype=bool)
        self.current_trace: float = self.trace_J
        self.step_count: int = 0

        # -- precompute adjacency matrices for GNN --
        # Signed correlation matrix (zero diagonal)
        diag_sqrt = np.sqrt(np.abs(self.diag_J))
        denom = diag_sqrt[:, None] * diag_sqrt[None, :]
        self._corr: np.ndarray = J / (denom + 1e-12)
        np.fill_diagonal(self._corr, 0.0)             # remove self-loops

        # Standard (unsigned, row-stochastic) adjacency using |corr|
        # so that the row sums are always positive.
        abs_corr = np.abs(self._corr)
        row_sum_abs = abs_corr.sum(axis=1, keepdims=True)
        # Normalised adjacency (row-stochastic w.r.t. |corr|, zero diagonal)
        self.adj_norm: np.ndarray = abs_corr / (row_sum_abs + 1e-12)

        # Signed split adjacency for the signed_adj GNN variant.
        # adj_pos[i,j] = max(corr[i,j], 0) / row_sum_abs[i]
        # adj_neg[i,j] = max(-corr[i,j], 0) / row_sum_abs[i]
        corr_pos = np.maximum(self._corr, 0.0)
        corr_neg = np.maximum(-self._corr, 0.0)
        self.adj_pos: np.ndarray = corr_pos / (row_sum_abs + 1e-12)
        self.adj_neg: np.ndarray = corr_neg / (row_sum_abs + 1e-12)

        # Precompute single-sensor marginal gains for node-feature 4
        self._single_gains: np.ndarray = self._compute_single_gains()

    # --------------------------------------------------------------------------
    # Core environment API
    # --------------------------------------------------------------------------

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

        # -- Reward shaping ---------------------------------------------------
        # +delta_trace/trace_J  : normalised trace reduction (greedy submodular gain)
        # -1/N                  : small per-sensor cost (encourages parsimony)
        # +large bonus           : when constraint first satisfied
        delta_trace = prev_trace - self.current_trace          # >= 0
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

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _compute_posterior_trace(self, selected_indices: np.ndarray) -> float:
        """Compute trace( J - J*W*(W*J*W + sigma^2*I)^{-1}*W*J ) for the given selection.

        Uses the matrix-inversion identity:
            trace_posterior = trace(J) - trace( (J_SS + sigma^2*I)^{-1} * J_S^T * J_S )
        where J_S = J[:, S]  and  J_SS = J[S, :][:, S].

        The regularised version of J is used when the original is not PD.
        """
        if len(selected_indices) == 0:
            return self.trace_J

        S = selected_indices
        J_use = self._J_reg
        J_S = J_use[:, S]                             # N x |S|
        J_SS = J_use[np.ix_(S, S)]                   # |S| x |S|
        M = J_SS + (self.sigma ** 2) * np.eye(len(S)) # |S| x |S|

        # Stable solve rather than explicit inversion
        try:
            reduction = np.trace(J_S.T @ np.linalg.solve(M, J_S.T).T)
        except np.linalg.LinAlgError:
            reduction = np.trace(J_S @ np.linalg.lstsq(M, J_S.T, rcond=None)[0])

        return float(self.trace_J - reduction)

    def _compute_single_gains(self) -> np.ndarray:
        """Precompute the single-sensor marginal gain for every node.

        gain[i] = trace_J - posterior_trace({i}), normalised to [0, 1]
        by dividing by trace_J.  Used as an informative node feature.
        """
        gains = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            gains[i] = self.trace_J - self._compute_posterior_trace(np.array([i]))
        max_gain = gains.max()
        if max_gain > 1e-12:
            gains /= max_gain
        return gains

    def _get_state(self) -> "SensorSelectionEnv.State":
        """Compute per-node features and return the full state tuple.

        Node feature vector (5 dimensions)
        ------------------------------------
        0 : is_selected  -- binary (0/1)
        1 : diag_J[i] / diag_J.max()  -- normalised prior variance
        2 : correlation_with_selected  -- mean |corr(i, j)| for j in S
        3 : current_trace / trace_J   -- global coverage progress (same for all nodes)
        4 : single-sensor marginal gain (normalised, precomputed)
        """
        node_features = np.zeros((self.N, 5), dtype=np.float32)

        node_features[:, 0] = self.selected.astype(np.float32)
        # Normalise by max diagonal to be scale-invariant across problem instances
        diag_max = self.diag_J.max()
        node_features[:, 1] = (self.diag_J / (diag_max + 1e-12)).astype(np.float32)

        # Feature 2: mean of absolute correlations with currently selected set
        selected_indices = np.where(self.selected)[0]
        if len(selected_indices) > 0:
            corr_with_selected = np.abs(self._corr[:, selected_indices]).sum(axis=1)
            corr_with_selected /= (len(selected_indices) + 1e-12)
            node_features[:, 2] = corr_with_selected.astype(np.float32)

        # Feature 3: global trace progress (broadcast scalar)
        node_features[:, 3] = float(self.current_trace / (self.trace_J + 1e-12))

        # Feature 4: single-sensor marginal gain (precomputed at construction)
        node_features[:, 4] = self._single_gains

        return node_features, self.adj_norm.astype(np.float32), self.current_trace, self.selected.copy()

    # --------------------------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------------------------

    @property
    def n_selected(self) -> int:
        return int(self.selected.sum())

    @property
    def is_satisfied(self) -> bool:
        return self.current_trace <= self.epsilon

    def action_mask(self) -> np.ndarray:
        """Boolean mask of shape (N,): True where action is valid (node not yet selected)."""
        return ~self.selected
