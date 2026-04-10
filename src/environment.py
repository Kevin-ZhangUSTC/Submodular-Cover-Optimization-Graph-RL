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

Scalability enhancements for large N (e.g. N=256)
--------------------------------------------------
band_radius : int (default 0)
    When > 0, the adjacency matrix is **band-limited**: only node pairs
    whose index distance is ≤ *band_radius* receive non-zero edges.
    This reduces GNN message-passing complexity from O(N²) to O(N · band_radius),
    which is critical for J₀ Toeplitz kernels where the dominant correlations
    are local (within ≈2 · length_scale positions).

step_penalty : float (default 1.0)
    Multiplier for the per-step sensor cost.  The reward per step includes a
    term ``-step_penalty / N``; increasing this to 2.0 gives the policy a
    stronger incentive to stop early and use fewer sensors.

period_hint : float (default 0.0)
    When > 0, two Fourier features cos(2π·i/period_hint) and
    sin(2π·i/period_hint) are appended to each node's feature vector
    (features 6 and 7).  For J₀ kernels the optimal sensor spacing is
    approximately 2.4 · length_scale (the first zero of J₀); passing that
    value as *period_hint* gives the GNN an explicit periodicity prior and
    reduces the search space from C(N, k) to ≈N×k options.
    When period_hint=0, features 6 and 7 are set to zero (neutral).

Node feature vector
-------------------
idx | description                                         | range
 0  | is_selected (binary)                                | {0, 1}
 1  | normalised prior variance diag(J)[i] / max(diag(J))| [0, 1]
 2  | mean |corr(i, j)| for j in selected set            | [0, 1]
 3  | current_trace / trace_J (global coverage progress) | [0, 1]
 4  | single-sensor marginal gain (normalised)            | [0, 1]
 5  | absolute position i / (N−1) (NEW)                  | [0, 1]
 6  | cos(2π·i / period_hint)  (0 when period_hint=0)    | [-1, 1]
 7  | sin(2π·i / period_hint)  (0 when period_hint=0)    | [-1, 1]

Total node_feat_dim = 8.
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
    band_radius : int
        When > 0, build a sparse (band-limited) adjacency: only node pairs
        with |i − j| ≤ band_radius get non-zero edges, reducing GNN
        complexity from O(N²) to O(N · band_radius).  0 = dense (default).
    step_penalty : float
        Per-step sensor cost multiplier.  Reward includes ``-step_penalty/N``
        per action; increasing from 1.0 to 2.0 creates stronger pressure to
        use fewer sensors.
    period_hint : float
        When > 0, add Fourier features cos(2π·i/period_hint) and
        sin(2π·i/period_hint) as node features 6 and 7.  For J₀ kernels,
        set to ``2.4 * length_scale`` (first zero of J₀) to inject the
        optimal spacing as a structural prior.  0 = no Fourier features
        (features 6 and 7 are zero).
    """

    # -- type aliases --
    NodeFeatures = np.ndarray   # shape (N, NODE_FEAT_DIM)
    State = Tuple[NodeFeatures, np.ndarray, float, np.ndarray]
    # (node_features, adj_norm, current_trace, selected_mask)

    #: Total number of node features emitted by ``_get_state()``.
    NODE_FEAT_DIM: int = 8

    def __init__(
        self,
        J: np.ndarray,
        sigma: float,
        epsilon: float,
        reg_min_eig: float = 1e-6,
        band_radius: int = 0,
        step_penalty: float = 1.0,
        period_hint: float = 0.0,
    ) -> None:
        assert J.ndim == 2 and J.shape[0] == J.shape[1], "J must be square"
        self.J: np.ndarray = J.copy()
        self.N: int = J.shape[0]
        self.sigma: float = float(sigma)
        self.epsilon: float = float(epsilon)
        self.band_radius: int = int(band_radius)
        self.step_penalty: float = float(step_penalty)
        self.period_hint: float = float(period_hint)

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

        abs_corr = np.abs(self._corr)

        # Apply band-limiting when band_radius > 0  (Plan A)
        # Only keep edges within |i - j| <= band_radius; zero the rest.
        # This reduces GNN complexity from O(N²) to O(N · band_radius).
        if self.band_radius > 0:
            i_idx = np.arange(self.N)[:, None]
            j_idx = np.arange(self.N)[None, :]
            outside_band = np.abs(i_idx - j_idx) > self.band_radius
            abs_corr_band = abs_corr.copy()
            abs_corr_band[outside_band] = 0.0
            corr_band = self._corr.copy()
            corr_band[outside_band] = 0.0
        else:
            abs_corr_band = abs_corr
            corr_band = self._corr

        row_sum_abs = abs_corr_band.sum(axis=1, keepdims=True)
        # Normalised adjacency (row-stochastic w.r.t. |corr|, zero diagonal)
        self.adj_norm: np.ndarray = abs_corr_band / (row_sum_abs + 1e-12)

        # Signed split adjacency for the signed_adj GNN variant.
        # adj_pos[i,j] = max(corr[i,j], 0) / row_sum_abs[i]
        # adj_neg[i,j] = max(-corr[i,j], 0) / row_sum_abs[i]
        corr_pos = np.maximum(corr_band, 0.0)
        corr_neg = np.maximum(-corr_band, 0.0)
        self.adj_pos: np.ndarray = corr_pos / (row_sum_abs + 1e-12)
        self.adj_neg: np.ndarray = corr_neg / (row_sum_abs + 1e-12)

        # Precompute position encodings (used in _get_state)  (Plan A, E)
        self._pos_enc: np.ndarray = (
            np.arange(self.N, dtype=np.float32) / max(self.N - 1, 1)
        )  # shape (N,), values in [0, 1]

        if self.period_hint > 0.0:
            angles = 2.0 * np.pi * np.arange(self.N) / self.period_hint
            self._fourier_cos: np.ndarray = np.cos(angles).astype(np.float32)
            self._fourier_sin: np.ndarray = np.sin(angles).astype(np.float32)
        else:
            self._fourier_cos = np.zeros(self.N, dtype=np.float32)
            self._fourier_sin = np.zeros(self.N, dtype=np.float32)

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
        # -step_penalty/N       : per-sensor cost (encourages parsimony; Plan C)
        # +large bonus           : when constraint first satisfied
        delta_trace = prev_trace - self.current_trace          # >= 0
        gain_reward = delta_trace / (self.trace_J + 1e-12)     # in [0, 1]
        cost_reward = -self.step_penalty / self.N

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
        """Compute the single-sensor marginal gain for every node using a closed form.

        For a single selected sensor at position i, the marginal trace reduction is:

            gain[i] = trace(J) − posterior_trace({i})
                    = ‖J[:, i]‖₂² / (J[i,i] + σ²)

        This closed form avoids N separate posterior-trace computations (which
        would be O(N³) in total via N linear solves), reducing initialisation to
        O(N²) even for large N (e.g. N=256).

        Gains are normalised to [0, 1] by dividing by the maximum gain.
        The regularised kernel matrix is used for numerical stability when J is
        not positive-definite (e.g. J₀ kernel).
        """
        J_use = self._J_reg
        # Vectorised closed form: gain[i] = ||J[:, i]||² / (J[i,i] + sigma²)
        col_sq_norms = np.sum(J_use ** 2, axis=0)          # (N,)
        denominators = self.diag_J + self.sigma ** 2        # (N,) - uses original J diagonal
        gains = col_sq_norms / (np.abs(denominators) + 1e-12)
        gains = gains.astype(np.float32)
        max_gain = gains.max()
        if max_gain > 1e-12:
            gains /= max_gain
        return gains

    def _get_state(self) -> "SensorSelectionEnv.State":
        """Compute per-node features and return the full state tuple.

        Node feature vector (8 dimensions)
        ------------------------------------
        0 : is_selected  -- binary (0/1)
        1 : diag_J[i] / diag_J.max()  -- normalised prior variance
        2 : correlation_with_selected  -- mean |corr(i, j)| for j in S
        3 : current_trace / trace_J   -- global coverage progress (same for all nodes)
        4 : single-sensor marginal gain (normalised, precomputed)
        5 : position i / (N−1)  -- absolute position embedding (Plan A)
        6 : cos(2π·i / period_hint)  -- Fourier spatial feature (Plan E; 0 if period_hint=0)
        7 : sin(2π·i / period_hint)  -- Fourier spatial feature (Plan E; 0 if period_hint=0)
        """
        node_features = np.zeros((self.N, self.NODE_FEAT_DIM), dtype=np.float32)

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

        # Feature 5: absolute position embedding (Plan A)
        node_features[:, 5] = self._pos_enc

        # Features 6-7: Fourier spatial features for J₀ periodicity (Plan E)
        node_features[:, 6] = self._fourier_cos
        node_features[:, 7] = self._fourier_sin

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
