"""
Brute-force optimal sensor-selection solver.

For small N (≤ 18) the minimum-size sensor set that satisfies

    trace( J - J_S (J_SS + σ²I)⁻¹ J_S^T ) ≤ ε

can be found by enumerating all 2^N subsets in order of increasing size.

This solver is used as a ground-truth baseline to prove that:
  1. The greedy algorithm is NOT always optimal (since the posterior trace
     function is non-submodular — see ``verify_non_submodularity``).
  2. A trained GNN+RL policy can learn to match or exceed this optimal
     on instances where greedy fails.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np


# ── Posterior trace helpers ───────────────────────────────────────────────────

def compute_posterior_trace(J: np.ndarray, sigma: float, S: List[int]) -> float:
    """Return tr(Σ_S) = tr(J) - tr(J_S (J_SS + σ²I)⁻¹ J_S^T).

    Parameters
    ----------
    J : np.ndarray
        N×N (regularised) covariance matrix.
    sigma : float
        Observation noise standard deviation.
    S : list of int
        Indices of selected sensors.

    Returns
    -------
    float
        Posterior trace value (≥ 0 for PD J).
    """
    if len(S) == 0:
        return float(np.trace(J))
    J_ss = J[np.ix_(S, S)]
    J_all_s = J[:, S]
    M = J_ss + sigma ** 2 * np.eye(len(S))
    try:
        reduction = np.trace(J_all_s @ np.linalg.solve(M, J_all_s.T))
    except np.linalg.LinAlgError:
        reduction = np.trace(J_all_s @ np.linalg.lstsq(M, J_all_s.T, rcond=None)[0])
    return float(np.trace(J) - reduction)


def marginal_gain(J: np.ndarray, sigma: float, S: set, i: int) -> float:
    """Reduction in posterior trace from adding sensor *i* to set *S*."""
    return compute_posterior_trace(J, sigma, list(S)) - \
           compute_posterior_trace(J, sigma, list(S | {i}))


# ── Brute-force optimal ───────────────────────────────────────────────────────

def find_optimal_sensor_set(
    J: np.ndarray,
    sigma: float,
    epsilon: float,
    max_n: int = 18,
) -> Tuple[List[int], int, bool]:
    """Find the minimum-size set S such that tr(Σ_S) ≤ epsilon.

    Enumerates subsets in order of increasing size (1, 2, 3, …) and returns
    the first feasible subset found.  Exponential in N — only use for N ≤ 18.

    Parameters
    ----------
    J : np.ndarray
        N×N covariance matrix.
    sigma : float
        Noise standard deviation.
    epsilon : float
        Trace threshold.
    max_n : int
        Safety cap; raises a warning and returns a heuristic result for N > max_n.

    Returns
    -------
    optimal_set : list of int
        Indices of selected sensors (length = optimal size).
    optimal_size : int
        Size of the optimal solution.
    solved_optimally : bool
        True if the result is provably optimal (i.e. N ≤ max_n).
    """
    N = J.shape[0]
    if N > max_n:
        warnings.warn(
            f"N={N} > max_n={max_n}: brute-force is infeasible. "
            "Returning a greedy approximation instead.",
            RuntimeWarning,
        )
        return _greedy_fallback(J, sigma, epsilon), -1, False

    # Special case: whole set satisfies?
    if compute_posterior_trace(J, sigma, list(range(N))) > epsilon:
        return list(range(N)), N, True  # unsatisfiable — return full set

    for k in range(0, N + 1):
        for combo in combinations(range(N), k):
            if compute_posterior_trace(J, sigma, list(combo)) <= epsilon:
                return list(combo), k, True

    return list(range(N)), N, True  # should never reach here


def _greedy_fallback(J: np.ndarray, sigma: float, epsilon: float) -> List[int]:
    """Simple greedy heuristic used when N > max_n for brute-force."""
    N = J.shape[0]
    selected: List[int] = []
    remaining = set(range(N))
    trace = float(np.trace(J))
    while trace > epsilon and remaining:
        best_i = max(remaining, key=lambda i: marginal_gain(J, sigma, set(selected), i))
        selected.append(best_i)
        remaining.remove(best_i)
        trace = compute_posterior_trace(J, sigma, selected)
    return selected


# ── Non-submodularity verification ───────────────────────────────────────────

def verify_non_submodularity(
    J: np.ndarray,
    sigma: float,
    n_checks: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    """Count submodularity violations in the marginal-gain function.

    Submodularity requires: for all A ⊆ B and i ∉ B,
        Δ(A, i) ≥ Δ(B, i)   (diminishing returns).

    A *violation* is a triplet (A, B, i) where Δ(A, i) < Δ(B, i).

    Returns
    -------
    violations : int
        Number of (A, B, i) triples that violate submodularity.
    checks : int
        Total number of triples checked.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = J.shape[0]
    violations = 0
    checks = 0
    for _ in range(n_checks):
        # Random A ⊆ B ⊆ {0..N-1}, i ∉ B
        universe = list(range(N))
        size_B = rng.integers(1, N)
        B_list = rng.choice(universe, size=size_B, replace=False).tolist()
        B = set(B_list)
        size_A = rng.integers(0, len(B_list) + 1)
        A = set(rng.choice(B_list, size=size_A, replace=False).tolist())
        outside = [j for j in universe if j not in B]
        if not outside:
            continue
        i = int(rng.choice(outside))
        dA = marginal_gain(J, sigma, A, i)
        dB = marginal_gain(J, sigma, B, i)
        checks += 1
        if dA < dB - 1e-9:
            violations += 1
    return violations, checks


# ── Greedy vs optimal gap analysis ───────────────────────────────────────────

def greedy_vs_optimal(
    J: np.ndarray,
    sigma: float,
    epsilon: float,
) -> dict:
    """Compare greedy and optimal solutions.

    Returns a dict with keys:
      - ``greedy_set``, ``greedy_size``
      - ``optimal_set``, ``optimal_size``
      - ``gap`` : greedy_size − optimal_size  (positive ↔ greedy is suboptimal)
      - ``greedy_trace``, ``optimal_trace``
    """
    optimal_set, optimal_size, solved = find_optimal_sensor_set(J, sigma, epsilon)
    greedy_set = _greedy_fallback(J, sigma, epsilon)
    greedy_size = len(greedy_set)
    return {
        "greedy_set": greedy_set,
        "greedy_size": greedy_size,
        "greedy_trace": compute_posterior_trace(J, sigma, greedy_set),
        "optimal_set": optimal_set,
        "optimal_size": optimal_size,
        "optimal_trace": compute_posterior_trace(J, sigma, optimal_set),
        "gap": greedy_size - optimal_size,
        "solved_optimally": solved,
    }
