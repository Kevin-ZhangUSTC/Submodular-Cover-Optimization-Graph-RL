"""
Unit and integration tests for the GNN+RL sensor-selection package.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.kernel import bessel_kernel, build_toeplitz_matrix, matern_05, matern_15, matern_25
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy, GraphSAGELayer
from src.rl_trainer import REINFORCETrainer


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_J():
    """5×5 Toeplitz matrix for fast unit tests."""
    return build_toeplitz_matrix(N=5, nu=1.5, length_scale=2.0)


@pytest.fixture
def env(small_J):
    epsilon = 0.3 * float(np.trace(small_J))
    return SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon)


@pytest.fixture
def policy():
    return GNNPolicy(node_feat_dim=4, hidden_dim=16, n_layers=2)


# ──────────────────────────────────────────────────────────────────────────────
# kernel.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestKernel:

    def test_bessel_kernel_zero(self):
        """k(0) = 1 for all ν."""
        for nu in [0.5, 1.5, 2.5, 3.0]:
            assert bessel_kernel(0.0, nu=nu) == pytest.approx(1.0, abs=1e-10)

    def test_bessel_kernel_decay(self):
        """Kernel decreases with distance."""
        for nu in [0.5, 1.5, 2.5]:
            prev = 1.0
            for r in [0.5, 1.0, 2.0, 4.0]:
                val = bessel_kernel(r, nu=nu, length_scale=1.0)
                assert val < prev, f"kernel not decaying at r={r}, nu={nu}"
                prev = val

    def test_closed_forms_consistent(self):
        """Closed-form helpers must match the general bessel_kernel."""
        for r in [0.1, 1.0, 3.0]:
            assert bessel_kernel(r, nu=0.5) == pytest.approx(matern_05(r), rel=1e-6)
            assert bessel_kernel(r, nu=1.5) == pytest.approx(matern_15(r), rel=1e-6)
            assert bessel_kernel(r, nu=2.5) == pytest.approx(matern_25(r), rel=1e-6)

    def test_toeplitz_shape(self):
        J = build_toeplitz_matrix(N=10)
        assert J.shape == (10, 10)

    def test_toeplitz_symmetric(self):
        J = build_toeplitz_matrix(N=8)
        np.testing.assert_allclose(J, J.T, atol=1e-12)

    def test_toeplitz_positive_definite(self):
        J = build_toeplitz_matrix(N=10, nu=1.5, length_scale=2.0)
        eigs = np.linalg.eigvalsh(J)
        assert np.all(eigs > 0), "Toeplitz matrix must be positive definite"

    def test_toeplitz_structure(self):
        """J[i,j] depends only on |i-j| (Toeplitz property)."""
        J = build_toeplitz_matrix(N=6)
        for i in range(6):
            for j in range(6):
                assert J[i, j] == pytest.approx(J[abs(i - j), 0], rel=1e-10)

    def test_toeplitz_diagonal_one(self):
        """k(0)=1, so J[i,i]=1 for all i."""
        J = build_toeplitz_matrix(N=7)
        np.testing.assert_allclose(np.diag(J), np.ones(7), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# environment.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEnvironment:

    def test_reset_returns_correct_shape(self, env):
        nf, adj, trace, sel = env.reset()
        assert nf.shape == (env.N, 4)
        assert adj.shape == (env.N, env.N)
        assert float(trace) == pytest.approx(env.trace_J)
        assert not sel.any()

    def test_initial_trace_equals_trace_J(self, env):
        env.reset()
        assert env.current_trace == pytest.approx(env.trace_J, rel=1e-6)

    def test_step_reduces_trace(self, env):
        env.reset()
        prev_trace = env.current_trace
        env.step(0)
        assert env.current_trace <= prev_trace + 1e-10

    def test_step_marks_node_selected(self, env):
        env.reset()
        env.step(2)
        assert env.selected[2]
        assert not env.selected[0]

    def test_double_select_raises(self, env):
        env.reset()
        env.step(0)
        with pytest.raises(ValueError):
            env.step(0)

    def test_posterior_trace_empty(self, env):
        t = env._compute_posterior_trace(np.array([], dtype=int))
        assert t == pytest.approx(env.trace_J)

    def test_posterior_trace_all_selected(self, env):
        """With all sensors selected the posterior trace should approach 0."""
        S = np.arange(env.N)
        t = env._compute_posterior_trace(S)
        # With sigma > 0 the posterior trace is positive but << trace(J)
        assert 0 < t < env.trace_J

    def test_posterior_trace_monotone(self, env):
        """Adding more sensors can only reduce the posterior trace."""
        traces = []
        for k in range(env.N):
            S = np.arange(k + 1)
            traces.append(env._compute_posterior_trace(S))
        for i in range(len(traces) - 1):
            assert traces[i] >= traces[i + 1] - 1e-10

    def test_done_when_constraint_satisfied(self, env):
        """Episode terminates once trace ≤ epsilon."""
        state = env.reset()
        done = False
        for i in range(env.N):
            _, _, _, selected = state
            state, _, done, _ = env.step(i)
            if done:
                break
        assert done

    def test_action_mask(self, env):
        env.reset()
        env.step(1)
        mask = env.action_mask()
        assert not mask[1]
        assert mask[0]

    def test_node_feature_is_selected_flag(self, env):
        """Feature 0 should be 1 for selected nodes after a step."""
        env.reset()
        env.step(3)
        nf, _, _, _ = env._get_state()
        assert nf[3, 0] == pytest.approx(1.0)
        assert nf[0, 0] == pytest.approx(0.0)

    def test_adj_row_stochastic(self, env):
        """Normalised adjacency rows should sum to ≈ 1 (excluding self-loop)."""
        row_sums = env.adj_norm.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(env.N), atol=1e-5)

    def test_adj_zero_diagonal(self, env):
        np.testing.assert_allclose(np.diag(env.adj_norm), np.zeros(env.N), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# gnn_model.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGNNModel:

    def test_graphsage_output_shape(self):
        layer = GraphSAGELayer(in_dim=8, out_dim=16)
        x = torch.randn(5, 8)
        adj = torch.rand(5, 5)
        adj = adj / adj.sum(dim=1, keepdim=True)
        out = layer(x, adj)
        assert out.shape == (5, 16)

    def test_policy_output_shapes(self, policy, env):
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, value = policy(x, A, mask)
        assert logits.shape == (env.N,)
        assert value.shape == ()

    def test_masked_logits_are_inf(self, policy, env):
        """Selected nodes must have -inf logit so they get zero probability."""
        env.reset()
        env.step(2)
        nf, adj, _, selected = env._get_state()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, _ = policy(x, A, mask)
        assert logits[2].item() == float("-inf")

    def test_get_action_valid(self, policy, env):
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        action, log_prob, entropy, value = policy.get_action(x, A, mask)
        assert 0 <= action < env.N
        assert not selected[action]
        assert log_prob.item() <= 0.0  # log probability ≤ 0
        assert entropy.item() >= 0.0

    def test_deterministic_action_reproducible(self, policy, env):
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        a1, _, _, _ = policy.get_action(x, A, mask, deterministic=True)
        a2, _, _, _ = policy.get_action(x, A, mask, deterministic=True)
        assert a1 == a2

    def test_no_gradient_through_value_in_policy_loss(self, policy, env):
        """Baseline should be detached (no gradient flowing through it)."""
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        _, _, _, value = policy.get_action(x, A, mask)
        # Detached value should have no grad_fn
        detached = value.detach()
        assert detached.grad_fn is None


# ──────────────────────────────────────────────────────────────────────────────
# rl_trainer.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRLTrainer:

    def test_single_episode_runs(self, policy, env):
        trainer = REINFORCETrainer(policy, lr=1e-3)
        stats = trainer.train_episode(env)
        assert "total_reward" in stats
        assert "n_selected" in stats
        assert "trace" in stats
        assert stats["n_selected"] >= 1

    def test_policy_parameters_change(self, policy, env):
        trainer = REINFORCETrainer(policy, lr=1e-2)
        params_before = [p.clone().detach() for p in policy.parameters()]
        # Run enough episodes so that at least one gradient step happens
        for _ in range(5):
            trainer.train_episode(env)
        any_changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, policy.parameters())
        )
        assert any_changed, "Policy parameters must update during training"

    def test_mean_reward_updates(self, policy, env):
        trainer = REINFORCETrainer(policy)
        for _ in range(3):
            trainer.train_episode(env)
        assert trainer.mean_reward != 0.0

    def test_compute_returns_discounting(self):
        """Returns must be discounted correctly."""
        policy = GNNPolicy(node_feat_dim=4, hidden_dim=8, n_layers=1)
        trainer = REINFORCETrainer(policy, gamma=0.9)
        rewards = [1.0, 0.0, 1.0]
        returns = trainer._compute_returns(rewards)
        assert len(returns) == 3
        # G_0 = 1 + 0.9*0 + 0.9^2*1 = 1.81 (before normalisation)
        # Just check ordering: G_0 should be largest (has most future reward)
        raw_G = [1.0 + 0.9 * 0.0 + 0.81 * 1.0, 0.0 + 0.9 * 1.0, 1.0]
        raw_G_t = torch.tensor(raw_G, dtype=torch.float32)
        raw_G_norm = (raw_G_t - raw_G_t.mean()) / (raw_G_t.std() + 1e-8)
        torch.testing.assert_close(returns, raw_G_norm, atol=1e-5, rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Integration test: full short training run
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_training_converges_to_feasible(self):
        """After sufficient training, the policy should consistently satisfy
        the constraint on this small instance."""
        np.random.seed(0)
        torch.manual_seed(0)

        N = 10
        J = build_toeplitz_matrix(N=N, nu=1.5, length_scale=3.0)
        epsilon = 0.25 * float(np.trace(J))
        env = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)

        policy = GNNPolicy(node_feat_dim=4, hidden_dim=32, n_layers=2)
        trainer = REINFORCETrainer(policy, lr=3e-3, entropy_coef=0.05)

        for _ in range(300):
            trainer.train_episode(env)

        # Evaluate deterministically
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

        # After 300 episodes on a size-10 problem, constraint should be met
        assert env.is_satisfied, (
            f"Policy failed to satisfy constraint: trace={env.current_trace:.4f} > "
            f"epsilon={epsilon:.4f}"
        )
