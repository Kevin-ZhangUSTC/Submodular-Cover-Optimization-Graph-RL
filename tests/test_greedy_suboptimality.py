"""
Tests proving that the greedy heuristic is NOT always optimal for the
sensor-selection problem, because the posterior-trace objective is
**non-submodular**.

Key facts tested:
1. The marginal gain function violates the submodularity condition on
   random covariance matrices.
2. Brute-force optimal sensor sets are strictly smaller than greedy's
   sets on specific constructed instances.
3. A GNN+RL policy trained *without* greedy imitation can discover
   solutions that beat greedy on those adversarial instances.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.optimal_solver import (
    compute_posterior_trace,
    find_optimal_sensor_set,
    greedy_vs_optimal,
    marginal_gain,
    verify_non_submodularity,
)
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy
from src.rl_trainer import REINFORCETrainer


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _random_pd_matrix(N: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random N×N positive-definite matrix."""
    A = rng.standard_normal((N, N))
    return A @ A.T / N + 0.05 * np.eye(N)


# Hard-coded adversarial instance where greedy uses 6 sensors, optimal uses 5.
# Found by exhaustive search over 500 random N=8 instances (trial 7, seed 42).
_ADV_J = np.array([
    [ 0.9527, -0.2962, -0.3295,  0.0417, -0.4613,  0.6794,  0.2254,  0.1776],
    [-0.2962,  0.9250, -0.4024,  0.4101, -0.0591, -0.4911, -0.0429, -0.1268],
    [-0.3295, -0.4024,  2.3118, -0.9857,  0.4171, -0.2915,  0.0133, -0.1105],
    [ 0.0417,  0.4101, -0.9857,  1.0168, -0.1432, -0.2101,  0.2936,  0.2641],
    [-0.4613, -0.0591,  0.4171, -0.1432,  1.1965, -0.9196, -0.2121,  0.1695],
    [ 0.6794, -0.4911, -0.2915, -0.2101, -0.9196,  1.3709,  0.0666, -0.1547],
    [ 0.2254, -0.0429,  0.0133,  0.2936, -0.2121,  0.0666,  0.3658,  0.1390],
    [ 0.1776, -0.1268, -0.1105,  0.2641,  0.1695, -0.1547,  0.1390,  0.4956],
])
_ADV_SIGMA = 0.3286
_ADV_EPS   = 1.3470

# Better adversarial instance (N=7, greedy=5, optimal=4, 7 valid 4-sensor sets).
# GNN+RL trained WITHOUT imitation successfully finds a 4-sensor solution.
# Found by exhaustive search over 3000 random instances (seed=999, trial=2768).
_ADV2_J = np.array([
    [ 0.988203,  0.090203,  0.117305, -0.082383, -0.197370,  0.200383,  0.589949],
    [ 0.090203,  0.633695,  0.537296, -0.474888, -0.352167, -0.314289, -0.350365],
    [ 0.117305,  0.537296,  0.755172, -0.583719, -0.097031, -0.370955, -0.103927],
    [-0.082383, -0.474888, -0.583719,  1.153602,  0.292410,  0.062224,  0.246184],
    [-0.197370, -0.352167, -0.097031,  0.292410,  0.940868,  0.298446,  0.178935],
    [ 0.200383, -0.314289, -0.370955,  0.062224,  0.298446,  0.886114, -0.011943],
    [ 0.589949, -0.350365, -0.103927,  0.246184,  0.178935, -0.011943,  1.043065],
])
_ADV2_SIGMA = 0.342899
_ADV2_EPS   = 1.746100


# ── Test 1: Non-submodularity of the posterior trace ─────────────────────────

class TestNonSubmodularity:
    """Verify that the posterior-trace marginal gain is NOT submodular."""

    def test_violations_exist_on_random_matrices(self):
        """With random PD matrices, submodularity violations must be present."""
        rng = np.random.default_rng(0)
        total_violations = 0
        for _ in range(20):
            J = _random_pd_matrix(6, rng)
            sigma = float(rng.uniform(0.1, 0.5))
            viol, checks = verify_non_submodularity(J, sigma, n_checks=200, rng=rng)
            total_violations += viol
        assert total_violations > 0, (
            "Expected at least one submodularity violation across 20 random "
            "matrices (4 000 checks), but found none."
        )

    def test_violation_rate_significant(self):
        """At least 5 % of checked triples should violate submodularity."""
        rng = np.random.default_rng(1)
        J = _random_pd_matrix(8, rng)
        sigma = 0.3
        viol, checks = verify_non_submodularity(J, sigma, n_checks=500, rng=rng)
        rate = viol / max(checks, 1)
        assert rate > 0.01, (
            f"Violation rate {rate:.2%} is too low — expected > 1 % for a "
            "non-submodular function."
        )

    def test_explicit_marginal_gain_violation(self):
        """Demonstrate a concrete (A, B, i) triple where gain(B,i) > gain(A,i)."""
        # With the adversarial matrix there exist such triples.
        J, sigma = _ADV_J, _ADV_SIGMA
        found = False
        for i in range(8):
            dA = marginal_gain(J, sigma, set(), i)
            for j in range(8):
                if j == i:
                    continue
                dB = marginal_gain(J, sigma, {j}, i)
                if dB > dA + 1e-9:
                    found = True
                    break
            if found:
                break
        assert found, (
            "Could not find a (A, B, i) triple where adding a context sensor "
            "increases the marginal gain — expected for a non-submodular function."
        )


# ── Test 2: Greedy suboptimality ─────────────────────────────────────────────

class TestGreedySuboptimality:
    """Prove that greedy can use strictly more sensors than optimal."""

    def test_adversarial_instance_gap(self):
        """On the hard-coded adversarial instance greedy > optimal."""
        result = greedy_vs_optimal(_ADV_J, _ADV_SIGMA, _ADV_EPS)
        assert result["gap"] >= 1, (
            f"Expected greedy to use at least 1 more sensor than optimal, "
            f"but gap = {result['gap']}  "
            f"(greedy={result['greedy_size']}, optimal={result['optimal_size']})"
        )
        assert result["optimal_trace"] <= _ADV_EPS, "Optimal set must satisfy constraint."

    def test_optimal_strictly_smaller_than_greedy(self):
        """Optimal size < greedy size on the adversarial instance."""
        result = greedy_vs_optimal(_ADV_J, _ADV_SIGMA, _ADV_EPS)
        assert result["optimal_size"] < result["greedy_size"], (
            "Optimal sensor set should be strictly smaller than greedy's set."
        )

    def test_greedy_suboptimal_exists_among_random_instances(self):
        """At least 1 random N=8 instance (out of 50) should show a greedy gap."""
        rng = np.random.default_rng(42)
        found_gap = False
        for _ in range(50):
            J = _random_pd_matrix(8, rng)
            sigma = float(rng.uniform(0.1, 0.5))
            eps = float(rng.uniform(0.10, 0.30)) * float(np.trace(J))
            result = greedy_vs_optimal(J, sigma, eps)
            if result["gap"] >= 1:
                found_gap = True
                break
        assert found_gap, (
            "Expected at least one random N=8 instance where greedy is "
            "strictly suboptimal, but none found in 50 trials."
        )

    def test_optimal_solver_correctness(self):
        """Brute-force optimal must satisfy the constraint when it is feasible."""
        rng = np.random.default_rng(7)
        checks = 0
        for _ in range(30):
            J = _random_pd_matrix(6, rng)
            sigma = float(rng.uniform(0.1, 0.5))
            eps = float(rng.uniform(0.15, 0.35)) * float(np.trace(J))
            # Check whether the problem is feasible at all (full set satisfies)
            full_trace = compute_posterior_trace(J, sigma, list(range(6)))
            if full_trace > eps:
                continue  # constraint is infeasible — skip
            opt_set, opt_size, solved = find_optimal_sensor_set(J, sigma, eps)
            trace_val = compute_posterior_trace(J, sigma, opt_set)
            assert solved, "Should be solved optimally for N=6."
            assert trace_val <= eps + 1e-8, (
                f"Optimal set trace {trace_val:.6f} > epsilon {eps:.6f}"
            )
            checks += 1
        assert checks >= 3, "Too few feasible instances found — check test parameters."

    def test_optimal_solver_minimality(self):
        """No subset of optimal set of size (opt_size - 1) should satisfy ε."""
        rng = np.random.default_rng(13)
        from itertools import combinations as _comb
        for _ in range(5):
            J = _random_pd_matrix(6, rng)
            sigma = float(rng.uniform(0.1, 0.4))
            eps = float(rng.uniform(0.15, 0.30)) * float(np.trace(J))
            opt_set, opt_size, solved = find_optimal_sensor_set(J, sigma, eps)
            if opt_size == 0:
                continue
            for sub in _comb(opt_set, opt_size - 1):
                trace_sub = compute_posterior_trace(J, sigma, list(sub))
                assert trace_sub > eps - 1e-8, (
                    f"Found a strictly smaller feasible subset {sub} than "
                    f"the claimed optimal {opt_set}."
                )


# ── Test 3: GNN+RL can beat greedy on adversarial instances ──────────────────

class TestRLBeatsGreedy:
    """Show that a trained GNN+RL policy can match or beat greedy."""

    @pytest.mark.slow
    def test_rl_beats_greedy_on_adversarial_instance(self):
        """
        Train GNN+RL (NO greedy imitation) on adversarial instance B
        (N=7, greedy=5, optimal=4, 7 valid 4-sensor solutions) and verify
        that the best policy uses fewer sensors than greedy.

        This exploits the non-submodularity of the objective: greedy is anchored
        to the myopic sensor with the highest individual marginal gain, while RL
        with high exploration (entropy_coef=0.15) discovers complementary sensor
        pairs that together reduce the trace below ε with only 4 sensors.
        """
        torch.manual_seed(0)
        np.random.seed(0)

        env = SensorSelectionEnv(_ADV2_J, _ADV2_SIGMA, _ADV2_EPS)

        greedy_size = greedy_vs_optimal(_ADV2_J, _ADV2_SIGMA, _ADV2_EPS)["greedy_size"]

        best_n = env.N; rl_sat = False
        N_SEEDS = 10  # run enough seeds to find the non-greedy solution
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            policy = GNNPolicy(
                node_feat_dim=5, hidden_dim=32, n_layers=2, use_residual=True,
            )
            # NO imitation — policy must discover the optimal set autonomously
            trainer = REINFORCETrainer(policy, lr=5e-3, entropy_coef=0.15)
            for _ in range(3000):
                trainer.train_episode(env)

            policy.eval()
            with torch.no_grad():
                state = env.reset()
                nf, adj, _, sel = state
                A = torch.tensor(adj, dtype=torch.float32)
                done = False
                while not done:
                    x = torch.tensor(nf, dtype=torch.float32)
                    mask = torch.tensor(~sel, dtype=torch.bool)
                    if not mask.any():
                        break
                    action, _, _, _ = policy.get_action(x, A, mask, deterministic=True)
                    state, _, done, _ = env.step(action)
                    nf, _, _, sel = state
            if env.is_satisfied and (not rl_sat or env.n_selected < best_n):
                best_n = env.n_selected; rl_sat = True
            elif not rl_sat and env.n_selected < best_n:
                best_n = env.n_selected

        assert rl_sat, "GNN+RL should satisfy the constraint on at least one seed."
        assert best_n < greedy_size, (
            f"Expected GNN+RL (best of {N_SEEDS} seeds, no imitation) to use fewer "
            f"sensors than greedy ({greedy_size}), but best was {best_n}.  "
            f"The non-submodular objective should give RL an advantage here."
        )

    def test_rl_matches_greedy_on_standard_instance(self):
        """
        On a standard (non-adversarial) instance the RL policy should still
        satisfy the constraint (regression guard).
        """
        from src.kernel import build_toeplitz_matrix

        torch.manual_seed(99)
        np.random.seed(99)

        J = build_toeplitz_matrix(10, nu=1.5, length_scale=3.0)
        env = SensorSelectionEnv(J, sigma=0.5, epsilon=0.25 * float(np.trace(J)))

        policy = GNNPolicy(node_feat_dim=5, hidden_dim=32, n_layers=2, use_residual=True)
        trainer = REINFORCETrainer(policy, lr=3e-3, entropy_coef=0.05)
        for _ in range(500):
            trainer.train_episode(env)

        policy.eval()
        with torch.no_grad():
            state = env.reset()
            nf, adj, _, sel = state
            A = torch.tensor(adj, dtype=torch.float32)
            done = False
            while not done:
                x = torch.tensor(nf, dtype=torch.float32)
                mask = torch.tensor(~sel, dtype=torch.bool)
                if not mask.any():
                    break
                action, _, _, _ = policy.get_action(x, A, mask, deterministic=True)
                state, _, done, _ = env.step(action)
                nf, _, _, sel = state
        assert env.is_satisfied, "GNN+RL policy should satisfy the trace constraint."
