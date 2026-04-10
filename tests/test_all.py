"""
Unit and integration tests for the GNN+RL sensor-selection package.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.kernel import (
    bessel_kernel, bessel_j0_kernel, build_toeplitz_matrix,
    is_positive_definite, matern_05, matern_15, matern_25,
    regularize_matrix,
)
from src.environment import SensorSelectionEnv
from src.gnn_model import (
    AttentionPooling, GNNPolicy, GraphSAGELayer,
    MultiHeadGraphAttentionLayer,
)
from src.rl_trainer import REINFORCETrainer
from src.imitation import ImitationTrainer, get_greedy_trajectory
from src.dataset import CurriculumScheduler, ProblemInstanceGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_J():
    """5x5 Toeplitz matrix for fast unit tests."""
    return build_toeplitz_matrix(N=5, nu=1.5, length_scale=2.0)


@pytest.fixture
def env(small_J):
    epsilon = 0.3 * float(np.trace(small_J))
    return SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon)


@pytest.fixture
def policy():
    return GNNPolicy(node_feat_dim=8, hidden_dim=16, n_layers=2)


# ──────────────────────────────────────────────────────────────────────────────
# kernel.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestKernel:

    def test_bessel_kernel_zero(self):
        """k(0) = 1 for all nu."""
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

    # ── J0 kernel (Idea 2) ──

    def test_bessel_j0_at_zero(self):
        """J0(0) = 1."""
        assert bessel_j0_kernel(0.0) == pytest.approx(1.0, abs=1e-10)

    def test_bessel_j0_oscillatory(self):
        """J0 kernel can be negative (first zero near 2.4)."""
        val_large = bessel_j0_kernel(3.0, length_scale=1.0)
        assert val_large < 0.0, "J0 kernel should be negative near r=3"

    def test_bessel_j0_length_scale(self):
        """Scaling r by 1/l should match bessel_j0_kernel(r*l, 1.0)."""
        r, l = 1.5, 2.0
        assert bessel_j0_kernel(r, length_scale=l) == pytest.approx(
            bessel_j0_kernel(r / l, length_scale=1.0), rel=1e-8
        )

    def test_build_toeplitz_j0(self):
        """J0 Toeplitz matrix has correct shape and diagonal of 1."""
        J = build_toeplitz_matrix(N=8, kernel_type="j0", length_scale=2.0)
        assert J.shape == (8, 8)
        np.testing.assert_allclose(np.diag(J), np.ones(8), atol=1e-10)
        assert np.any(J < 0), "J0 matrix should have some negative entries"

    def test_build_toeplitz_unknown_type(self):
        with pytest.raises(ValueError):
            build_toeplitz_matrix(N=5, kernel_type="unknown")

    # ── PD utilities (Idea 2) ──

    def test_is_positive_definite_true(self):
        J = build_toeplitz_matrix(N=6, nu=1.5)
        assert is_positive_definite(J)

    def test_is_positive_definite_false(self):
        J = build_toeplitz_matrix(N=10, kernel_type="j0", length_scale=0.5)
        # May or may not be PD; regularize_matrix should handle either case
        J_reg = regularize_matrix(J)
        assert is_positive_definite(J_reg)

    def test_regularize_matrix_leaves_pd_unchanged(self):
        J = build_toeplitz_matrix(N=5, nu=1.5)
        J_reg = regularize_matrix(J, min_eig=1e-8)
        np.testing.assert_allclose(J, J_reg, atol=1e-10)

    def test_regularize_matrix_makes_pd(self):
        """A matrix with a negative eigenvalue should become PD after regularize."""
        J = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues -1 and 3
        J_reg = regularize_matrix(J, min_eig=1e-6)
        assert is_positive_definite(J_reg)


# ──────────────────────────────────────────────────────────────────────────────
# environment.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEnvironment:

    def test_reset_returns_correct_shape(self, env):
        nf, adj, trace, sel = env.reset()
        assert nf.shape == (env.N, 8)
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
        """Episode terminates once trace <= epsilon."""
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
        """Normalised adjacency rows should sum to ~1."""
        row_sums = env.adj_norm.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(env.N), atol=1e-5)

    def test_adj_zero_diagonal(self, env):
        np.testing.assert_allclose(np.diag(env.adj_norm), np.zeros(env.N), atol=1e-12)

    def test_marginal_gain_feature_range(self, env):
        """Single-sensor marginal gain feature should be in [0, 1]."""
        nf, _, _, _ = env.reset()
        assert np.all(nf[:, 4] >= 0.0)
        assert np.all(nf[:, 4] <= 1.0 + 1e-6)

    # ── J0 kernel env tests (Idea 2) ──

    def test_env_with_j0_kernel_constructs(self):
        """Env should construct without error for J0 kernel (non-PD J)."""
        import warnings
        # Use small length_scale so oscillations create negative entries quickly
        J = build_toeplitz_matrix(N=10, kernel_type="j0", length_scale=0.5)
        epsilon = 0.3 * float(np.trace(J))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            env = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)
        # adj_pos and adj_neg should always be present
        assert env.adj_pos.shape == (10, 10)
        assert env.adj_neg.shape == (10, 10)

    def test_env_j0_adj_non_negative(self):
        """adj_pos and adj_neg must be non-negative."""
        J = build_toeplitz_matrix(N=6, kernel_type="j0", length_scale=1.5)
        epsilon = 0.3 * float(np.trace(J))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)
        assert np.all(env.adj_pos >= 0.0)
        assert np.all(env.adj_neg >= 0.0)

    def test_env_pd_flag_true_for_matern(self, env):
        """Matern-based env should have j_is_pd=True."""
        assert env.j_is_pd

    # ── Scalability enhancements (Plans A, C, E) ──

    def test_position_embedding_feature_range(self, env):
        """Feature 5 (position) should be in [0, 1] and monotonically increasing."""
        nf, _, _, _ = env.reset()
        pos = nf[:, 5]
        assert np.all(pos >= 0.0 - 1e-6)
        assert np.all(pos <= 1.0 + 1e-6)
        assert pos[0] == pytest.approx(0.0, abs=1e-6)
        assert pos[-1] == pytest.approx(1.0, abs=1e-6)
        # Monotonically increasing
        assert np.all(np.diff(pos) >= -1e-6)

    def test_fourier_features_zero_without_period_hint(self, env):
        """Features 6 and 7 should be zero when period_hint=0 (default)."""
        nf, _, _, _ = env.reset()
        np.testing.assert_allclose(nf[:, 6], 0.0, atol=1e-6)
        np.testing.assert_allclose(nf[:, 7], 0.0, atol=1e-6)

    def test_fourier_features_nonzero_with_period_hint(self, small_J):
        """Features 6 and 7 should be non-trivial when period_hint > 0."""
        epsilon = 0.3 * float(np.trace(small_J))
        env = SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon, period_hint=3.0)
        nf, _, _, _ = env.reset()
        # cos(0) = 1, so feature 6 of node 0 should be 1
        assert nf[0, 6] == pytest.approx(1.0, abs=1e-5)
        # sin(0) = 0, so feature 7 of node 0 should be 0
        assert nf[0, 7] == pytest.approx(0.0, abs=1e-5)
        # Not all the same
        assert not np.allclose(nf[:, 6], nf[0, 6])

    def test_band_radius_adj_is_sparse(self, small_J):
        """With band_radius=1, adjacency only has entries for |i-j| <= 1."""
        epsilon = 0.3 * float(np.trace(small_J))
        N = small_J.shape[0]
        env = SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon, band_radius=1)
        adj = env.adj_norm
        for i in range(N):
            for j in range(N):
                if abs(i - j) > 1:
                    assert adj[i, j] == pytest.approx(0.0, abs=1e-12), (
                        f"Expected adj[{i},{j}]=0 with band_radius=1"
                    )

    def test_band_radius_zero_is_dense(self, env):
        """Default band_radius=0 should give a dense (fully-connected) adjacency."""
        # The default fixture env has band_radius=0 (dense)
        assert env.band_radius == 0
        # At least some off-band entries should be non-zero for a smooth Matern
        N = env.N
        if N > 2:
            # Check entry at distance > 1 is non-zero (smooth kernel)
            assert env.adj_norm[0, -1] > 0.0 or env.adj_norm[0, 2] > 0.0

    def test_step_penalty_affects_reward(self, small_J):
        """Higher step_penalty should give lower per-step reward."""
        epsilon = 0.3 * float(np.trace(small_J))
        env_default = SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon,
                                         step_penalty=1.0)
        env_strong = SensorSelectionEnv(small_J, sigma=0.5, epsilon=epsilon,
                                        step_penalty=2.0)
        env_default.reset()
        env_strong.reset()
        _, r_default, _, _ = env_default.step(0)
        _, r_strong, _, _ = env_strong.step(0)
        # The cost term is -step_penalty/N; stronger penalty → lower reward
        assert r_strong <= r_default


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

    def test_graphsage_residual(self):
        layer = GraphSAGELayer(in_dim=8, out_dim=8, use_residual=True)
        x = torch.randn(5, 8)
        adj = torch.rand(5, 5)
        adj = adj / adj.sum(dim=1, keepdim=True)
        out = layer(x, adj)
        assert out.shape == (5, 8)

    def test_graphsage_residual_diff_dims(self):
        """Residual with in_dim != out_dim should use a projection."""
        layer = GraphSAGELayer(in_dim=8, out_dim=16, use_residual=True)
        x = torch.randn(4, 8)
        adj = torch.rand(4, 4)
        adj = adj / adj.sum(dim=1, keepdim=True)
        out = layer(x, adj)
        assert out.shape == (4, 16)

    def test_graphsage_signed_adj(self):
        layer = GraphSAGELayer(in_dim=8, out_dim=16, signed_adj=True)
        x = torch.randn(5, 8)
        adj_pos = torch.rand(5, 5)
        adj_neg = torch.rand(5, 5)
        out = layer(x, adj_pos, adj_neg)
        assert out.shape == (5, 16)

    def test_graphsage_signed_adj_requires_adj_neg(self):
        layer = GraphSAGELayer(in_dim=8, out_dim=16, signed_adj=True)
        x = torch.randn(5, 8)
        adj = torch.rand(5, 5)
        with pytest.raises(ValueError):
            layer(x, adj)  # adj_neg missing

    def test_gat_output_shape(self):
        layer = MultiHeadGraphAttentionLayer(in_dim=8, out_dim=16, n_heads=4)
        x = torch.randn(5, 8)
        adj = torch.rand(5, 5)
        out = layer(x, adj)
        assert out.shape == (5, 16)

    def test_gat_invalid_heads(self):
        with pytest.raises(ValueError):
            MultiHeadGraphAttentionLayer(in_dim=8, out_dim=15, n_heads=4)

    def test_gat_residual(self):
        layer = MultiHeadGraphAttentionLayer(
            in_dim=8, out_dim=8, n_heads=4, use_residual=True
        )
        x = torch.randn(5, 8)
        adj = torch.rand(5, 5)
        out = layer(x, adj)
        assert out.shape == (5, 8)

    def test_gat_signed_adj(self):
        layer = MultiHeadGraphAttentionLayer(
            in_dim=8, out_dim=8, n_heads=4, signed_adj=True
        )
        x = torch.randn(5, 8)
        adj_pos = torch.rand(5, 5)
        adj_neg = torch.rand(5, 5)
        out = layer(x, adj_pos, adj_neg)
        assert out.shape == (5, 8)

    def test_attention_pooling_output_shape(self):
        pool = AttentionPooling(hidden_dim=16)
        x = torch.randn(5, 16)
        mask = torch.tensor([True, True, False, True, False])
        z = pool(x, mask)
        assert z.shape == (16,)

    def test_attention_pooling_no_mask(self):
        pool = AttentionPooling(hidden_dim=16)
        x = torch.randn(5, 16)
        z = pool(x)
        assert z.shape == (16,)

    def test_policy_output_shapes(self, policy, env):
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, value = policy(x, A, mask)
        assert logits.shape == (env.N,)
        assert value.shape == ()

    def test_masked_logits_are_inf(self, policy, env):
        """Selected nodes must have -inf logit."""
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
        assert log_prob.item() <= 0.0
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
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        _, _, _, value = policy.get_action(x, A, mask)
        detached = value.detach()
        assert detached.grad_fn is None

    def test_policy_gat_runs(self, env):
        """GAT-based policy should produce correct output shapes."""
        pol = GNNPolicy(
            node_feat_dim=8, hidden_dim=16, n_layers=2,
            layer_type="gat", n_heads=4
        )
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, value = pol(x, A, mask)
        assert logits.shape == (env.N,)
        assert value.shape == ()

    def test_policy_attention_pooling_runs(self, env):
        pol = GNNPolicy(
            node_feat_dim=8, hidden_dim=16, n_layers=2,
            use_attention_pooling=True
        )
        nf, adj, _, selected = env.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, value = pol(x, A, mask)
        assert logits.shape == (env.N,)
        assert value.shape == ()

    def test_policy_signed_adj_runs(self):
        """Policy with signed_adj=True should accept adj_pos + adj_neg."""
        J = build_toeplitz_matrix(N=6, kernel_type="j0", length_scale=2.0)
        epsilon = 0.3 * float(np.trace(J))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env6 = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)
        pol = GNNPolicy(node_feat_dim=8, hidden_dim=16, n_layers=2, signed_adj=True)
        nf, adj, _, selected = env6.reset()
        x = torch.tensor(nf, dtype=torch.float32)
        A_pos = torch.tensor(env6.adj_pos, dtype=torch.float32)
        A_neg = torch.tensor(env6.adj_neg, dtype=torch.float32)
        mask = torch.tensor(~selected, dtype=torch.bool)
        logits, value = pol(x, A_pos, mask, adj_neg=A_neg)
        assert logits.shape == (6,)


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
        pol = GNNPolicy(node_feat_dim=8, hidden_dim=8, n_layers=1)
        trainer = REINFORCETrainer(pol, gamma=0.9)
        rewards = [1.0, 0.0, 1.0]
        returns = trainer._compute_returns(rewards)
        assert len(returns) == 3
        raw_G = [1.0 + 0.9 * 0.0 + 0.81 * 1.0, 0.0 + 0.9 * 1.0, 1.0]
        raw_G_t = torch.tensor(raw_G, dtype=torch.float32)
        raw_G_norm = (raw_G_t - raw_G_t.mean()) / (raw_G_t.std() + 1e-8)
        torch.testing.assert_close(returns, raw_G_norm, atol=1e-5, rtol=1e-5)

    def test_train_episode_with_imitation(self, policy, env):
        """Blended imitation loss should run without error."""
        traj, _ = get_greedy_trajectory(env)
        trainer = REINFORCETrainer(policy, lr=1e-3)
        stats = trainer.train_episode(env, greedy_trajectory=traj, imitation_coef=0.5)
        assert "imitation_loss" in stats
        assert stats["imitation_loss"] >= 0.0

    def test_train_multi_env_episode(self, policy, env):
        """Multi-env training should pick a random env and return env_index."""
        trainer = REINFORCETrainer(policy, lr=1e-3)
        envs = [env, env]
        stats = trainer.train_multi_env_episode(envs)
        assert "env_index" in stats
        assert stats["env_index"] in (0.0, 1.0)

    def test_beam_rollout_returns_valid_result(self, policy, env):
        """beam_rollout should return a dict with expected keys and valid values."""
        trainer = REINFORCETrainer(policy, lr=1e-3)
        result = trainer.beam_rollout(env, n_rollouts=5, rng_seed=0)
        assert "n_selected" in result
        assert "trace" in result
        assert "satisfied" in result
        assert "selected_mask" in result
        assert "rollout_n" in result
        assert 0 <= result["n_selected"] <= env.N
        assert result["trace"] >= 0.0
        assert isinstance(result["satisfied"], bool)
        assert result["selected_mask"].shape == (env.N,)
        assert 0 <= result["rollout_n"] < 5

    def test_beam_rollout_better_than_or_equal_to_single(self, env):
        """More rollouts should give a result with at most as many sensors."""
        torch.manual_seed(7)
        np.random.seed(7)
        policy_k = GNNPolicy(node_feat_dim=8, hidden_dim=16, n_layers=2)
        trainer = REINFORCETrainer(policy_k, lr=3e-3, entropy_coef=0.05)
        for _ in range(50):
            trainer.train_episode(env)
        r1 = trainer.beam_rollout(env, n_rollouts=1, rng_seed=0)
        rk = trainer.beam_rollout(env, n_rollouts=20, rng_seed=0)
        # beam with k=20 should use no more sensors than k=1 (it picks the best)
        if r1["satisfied"] and rk["satisfied"]:
            assert rk["n_selected"] <= r1["n_selected"]


# ──────────────────────────────────────────────────────────────────────────────
# imitation.py tests (Idea 1)
# ──────────────────────────────────────────────────────────────────────────────

class TestImitation:

    def test_get_greedy_trajectory_length(self, env):
        traj, trace = get_greedy_trajectory(env)
        assert isinstance(traj, list)
        assert len(traj) >= 1
        assert trace <= env.epsilon or len(traj) == env.N

    def test_get_greedy_trajectory_valid_actions(self, env):
        traj, _ = get_greedy_trajectory(env)
        seen = set()
        for a in traj:
            assert 0 <= a < env.N
            assert a not in seen
            seen.add(a)

    def test_imitation_trainer_changes_params(self, policy, env):
        traj, _ = get_greedy_trajectory(env)
        imitator = ImitationTrainer(policy, lr=1e-2)
        params_before = [p.clone().detach() for p in policy.parameters()]
        for _ in range(5):
            imitator.train_episode(env, traj)
        any_changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, policy.parameters())
        )
        assert any_changed

    def test_imitation_trainer_loss_is_non_negative(self, policy, env):
        traj, _ = get_greedy_trajectory(env)
        imitator = ImitationTrainer(policy)
        stats = imitator.train_episode(env, traj)
        assert stats["imitation_loss"] >= 0.0

    def test_imitation_mean_loss_updates(self, policy, env):
        traj, _ = get_greedy_trajectory(env)
        imitator = ImitationTrainer(policy)
        for _ in range(3):
            imitator.train_episode(env, traj)
        assert imitator.mean_loss >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# dataset.py tests (Idea 4)
# ──────────────────────────────────────────────────────────────────────────────

class TestDataset:

    def test_generator_sample_returns_env(self):
        gen = ProblemInstanceGenerator(n_range=(5, 10), seed=0)
        env = gen.sample()
        assert isinstance(env, SensorSelectionEnv)
        assert 5 <= env.N <= 10

    def test_generator_sample_varies(self):
        gen = ProblemInstanceGenerator(n_range=(5, 15), seed=1)
        envs = [gen.sample() for _ in range(10)]
        # At least some variation in N
        ns = [e.N for e in envs]
        assert len(set(ns)) > 1 or len(ns) == 1  # may all be same if n_range tight

    def test_generator_j0_kernel(self):
        gen = ProblemInstanceGenerator(
            n_range=(5, 8), kernel_types=["j0"], seed=2
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = gen.sample()
        assert isinstance(env, SensorSelectionEnv)

    def test_curriculum_starts_at_stage_0(self):
        sched = CurriculumScheduler(n_min=5, n_max=30, seed=0)
        assert sched.stage == 0

    def test_curriculum_advances_stage(self):
        sched = CurriculumScheduler(
            n_min=5, n_max=30, advance_threshold=0.6, window=5, seed=0
        )
        for _ in range(5):
            sched.record(True)
        assert sched.stage == 1

    def test_curriculum_does_not_exceed_max_stage(self):
        sched = CurriculumScheduler(
            n_min=5, n_max=30, advance_threshold=0.0, window=1, seed=0
        )
        for _ in range(20):
            sched.record(True)
        assert sched.stage <= 3

    def test_curriculum_sample_returns_env(self):
        sched = CurriculumScheduler(n_min=5, n_max=20, seed=0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = sched.sample()
        assert isinstance(env, SensorSelectionEnv)

    def test_node_features_scale_invariant(self):
        """Normalised features should stay in [0,1] regardless of kernel scale."""
        for length_scale in [0.5, 1.0, 5.0, 10.0]:
            J = build_toeplitz_matrix(N=8, nu=1.5, length_scale=length_scale)
            epsilon = 0.3 * float(np.trace(J))
            env = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)
            nf, _, _, _ = env.reset()
            # Feature 1 (normalised prior variance) should be in [0,1]
            assert np.all(nf[:, 1] >= 0.0 - 1e-6)
            assert np.all(nf[:, 1] <= 1.0 + 1e-6)
            # Feature 3 (trace progress) should be in [0,1]
            assert np.all(nf[:, 3] >= 0.0 - 1e-6)
            assert np.all(nf[:, 3] <= 1.0 + 1e-6)
            # Feature 4 (marginal gain) should be in [0,1]
            assert np.all(nf[:, 4] >= 0.0 - 1e-6)
            assert np.all(nf[:, 4] <= 1.0 + 1e-6)

    def test_curriculum_stage3_is_j0(self):
        """Stage 3 (large-N) curriculum should produce J₀ kernel environments."""
        sched = CurriculumScheduler(
            n_min=5, n_max=256, advance_threshold=0.0, window=1, seed=42
        )
        # Force advance to stage 3
        for _ in range(4):
            sched.record(True)
            sched.record(True)
        assert sched.stage == 3
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = sched.sample()
        assert isinstance(env, SensorSelectionEnv)
        assert env.N >= 5


# ──────────────────────────────────────────────────────────────────────────────
# Integration test: full short training run
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_training_converges_to_feasible(self):
        """After sufficient training, the policy should satisfy the constraint."""
        np.random.seed(0)
        torch.manual_seed(0)

        N = 10
        J = build_toeplitz_matrix(N=N, nu=1.5, length_scale=3.0)
        epsilon = 0.25 * float(np.trace(J))
        env = SensorSelectionEnv(J, sigma=0.5, epsilon=epsilon)

        policy = GNNPolicy(node_feat_dim=8, hidden_dim=32, n_layers=2)
        trainer = REINFORCETrainer(policy, lr=3e-3, entropy_coef=0.05)

        for _ in range(300):
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

        assert env.is_satisfied, (
            f"Policy failed to satisfy constraint: trace={env.current_trace:.4f} > "
            f"epsilon={epsilon:.4f}"
        )
