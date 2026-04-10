"""
Main training script for the GNN+RL submodular cover sensor-selection problem.

Usage
-----
    python main.py [--N 20] [--sigma 0.5] [--eps_frac 0.20]
                   [--kernel_type matern] [--nu 1.5] [--length_scale 3.0]
                   [--episodes 2000] [--lr 3e-4] [--seed 42]
                   [--layer_type sage] [--n_heads 4]
                   [--use_residual] [--use_attention_pooling]
                   [--band_radius 0] [--step_penalty 1.0]
                   [--beam_width 1] [--period_hint 0.0]
                   [--multi_env] [--n_train_envs 50]
                   [--n_min 5] [--n_max 30] [--curriculum]
                   [--no-plot]

The script will:
  1. Build the kernel matrix J (Matern or J0, or load from .mat).
  2. Run the greedy baseline for comparison.
  3. Train the GNN policy with REINFORCE + Actor-Critic baseline.
  4. Save the trained model to ``checkpoint.pt``.
  5. Plot the training curves to ``training_curve.png``.
"""

from __future__ import annotations

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")                         # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio

import config
from src.kernel import build_toeplitz_matrix
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy
from src.rl_trainer import REINFORCETrainer
from src.optimal_solver import get_greedy_trajectory
from src.dataset import ProblemInstanceGenerator, CurriculumScheduler


# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN+RL Sensor Selection")
    # Problem
    p.add_argument("--N", type=int, default=config.N)
    p.add_argument("--sigma", type=float, default=config.SIGMA)
    p.add_argument("--eps_frac", type=float, default=config.EPSILON_FRAC)
    p.add_argument("--kernel_type", type=str, default=config.KERNEL_TYPE,
                   choices=["matern", "j0"])
    p.add_argument("--nu", type=float, default=config.KERNEL_NU)
    p.add_argument("--length_scale", type=float, default=config.KERNEL_LENGTH_SCALE)
    # Training
    p.add_argument("--episodes", type=int, default=config.N_TRAIN_EPISODES)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    # GNN architecture
    p.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
    p.add_argument("--n_layers", type=int, default=config.N_GNN_LAYERS)
    p.add_argument("--layer_type", type=str, default=config.LAYER_TYPE,
                   choices=["sage", "gat"])
    p.add_argument("--n_heads", type=int, default=config.N_HEADS)
    p.add_argument("--attention_dropout", type=float, default=config.ATTENTION_DROPOUT)
    p.add_argument("--use_residual", action="store_true",
                   default=config.USE_RESIDUAL)
    p.add_argument("--use_attention_pooling", action="store_true",
                   default=config.USE_ATTENTION_POOLING)
    # Scalability enhancements for large N / J₀ kernels
    p.add_argument("--band_radius", type=int, default=config.BAND_RADIUS,
                   help="Sparse adjacency band radius (0=dense, >0=O(N·r) edges)")
    p.add_argument("--step_penalty", type=float, default=config.STEP_PENALTY,
                   help="Per-step sensor cost multiplier (default 1.0; try 2.0)")
    p.add_argument("--beam_width", type=int, default=config.BEAM_WIDTH,
                   help="Stochastic rollouts at eval time (1=greedy, >1=beam)")
    p.add_argument("--period_hint", type=float, default=config.PERIOD_HINT,
                   help="J₀ period hint for Fourier features (0=disabled; "
                        "recommended: 2.4 * length_scale)")
    # Multi-environment / curriculum
    p.add_argument("--multi_env", action="store_true", default=config.MULTI_ENV)
    p.add_argument("--n_train_envs", type=int, default=config.N_TRAIN_ENVS)
    p.add_argument("--n_min", type=int, default=config.N_MIN)
    p.add_argument("--n_max", type=int, default=config.N_MAX)
    p.add_argument("--curriculum", action="store_true", default=config.CURRICULUM)
    # Misc
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def build_primary_env(args: argparse.Namespace) -> SensorSelectionEnv:
    """Build the primary (single) training environment from CLI args."""
    MAT_FILE = "Kernal_-30.mat"
    MAT_VAR = "Kernal_exp"
    if os.path.exists(MAT_FILE) and args.kernel_type == "matern":
        mat = sio.loadmat(MAT_FILE)
        J_full = mat[MAT_VAR]
        J = np.array(J_full[:args.N, :args.N], dtype=float)
    else:
        J = build_toeplitz_matrix(
            args.N, nu=args.nu, length_scale=args.length_scale,
            kernel_type=args.kernel_type
        )
    epsilon = args.eps_frac * float(np.trace(J))
    return SensorSelectionEnv(
        J, sigma=args.sigma, epsilon=epsilon,
        band_radius=args.band_radius,
        step_penalty=args.step_penalty,
        period_hint=args.period_hint,
    )


def evaluate_policy(
    policy: GNNPolicy,
    env: SensorSelectionEnv,
    n_episodes: int = 1,
) -> dict:
    """Evaluate a trained policy deterministically."""
    policy.eval()
    results = []
    adj_neg_tensor = (
        torch.tensor(env.adj_neg, dtype=torch.float32)
        if policy.signed_adj
        else None
    )
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            node_features, adj, _, selected = state
            adj_tensor = torch.tensor(adj, dtype=torch.float32)
            done = False
            while not done:
                nf = torch.tensor(node_features, dtype=torch.float32)
                mask = torch.tensor(~selected, dtype=torch.bool)
                if not mask.any():
                    break
                action, _, _, _ = policy.get_action(
                    nf, adj_tensor, mask, deterministic=True,
                    adj_neg=adj_neg_tensor
                )
                state, _, done, _ = env.step(action)
                node_features, _, _, selected = state
            results.append({
                "n_selected": env.n_selected,
                "trace": env.current_trace,
                "satisfied": env.is_satisfied,
            })
    policy.train()
    return results[0]


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Automatically enable signed_adj for J0 kernel
    signed_adj = config.SIGNED_ADJ or (args.kernel_type == "j0")

    print(f"\n{'='*60}")
    print("  GNN + REINFORCE  -  Submodular Cover Sensor Selection")
    print(f"{'='*60}")
    print(f"  N={args.N}  sigma={args.sigma}  eps_frac={args.eps_frac}")
    print(f"  kernel={args.kernel_type}  nu={args.nu}  l={args.length_scale}")
    print(f"  layer={args.layer_type}  residual={args.use_residual}  "
          f"attn_pool={args.use_attention_pooling}  signed_adj={signed_adj}")
    print(f"  band_radius={args.band_radius}  step_penalty={args.step_penalty}  "
          f"period_hint={args.period_hint}  beam_width={args.beam_width}")

    # ── 1. Build primary environment ─────────────────────────────────────────
    env = build_primary_env(args)
    print(f"\n  trace(J) = {np.trace(env.J):.4f}   "
          f"epsilon = {env.epsilon:.4f}  j_is_pd={env.j_is_pd}")

    # ── 2. Greedy baseline ────────────────────────────────────────────────────
    greedy_traj, trace_greedy = get_greedy_trajectory(env)
    n_greedy = len(greedy_traj)
    print(f"\n  Greedy baseline : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f}, satisfied={trace_greedy <= env.epsilon})")

    # ── 3. Build model ────────────────────────────────────────────────────────
    policy = GNNPolicy(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        layer_type=args.layer_type,
        n_heads=args.n_heads,
        attention_dropout=args.attention_dropout,
        use_residual=args.use_residual,
        use_attention_pooling=args.use_attention_pooling,
        signed_adj=signed_adj,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  GNN parameters  : {n_params:,}")

    rl_trainer = REINFORCETrainer(
        policy,
        lr=args.lr,
        gamma=config.GAMMA,
        entropy_coef=config.ENTROPY_COEF,
        value_loss_coef=config.VALUE_LOSS_COEF,
    )

    # ── 4. Multi-env / curriculum setup ───────────────────────────────────────
    if args.multi_env:
        if args.curriculum:
            curriculum = CurriculumScheduler(
                n_min=args.n_min, n_max=args.n_max, seed=args.seed
            )
            sample_env = curriculum.sample
        else:
            generator = ProblemInstanceGenerator(
                n_range=(args.n_min, args.n_max),
                kernel_types=[args.kernel_type],
                seed=args.seed,
            )
            curriculum = None
            sample_env = generator.sample
        train_envs = [sample_env() for _ in range(args.n_train_envs)]
    else:
        train_envs = [env]
        curriculum = None

    # ── 5. RL training ────────────────────────────────────────────────────────
    print(f"\n  RL training for {args.episodes} episodes ...")

    all_rewards: list[float] = []
    all_lengths: list[float] = []
    all_satisfied: list[float] = []

    for episode in range(1, args.episodes + 1):
        if args.multi_env:
            stats = rl_trainer.train_multi_env_episode(train_envs)
            if curriculum is not None:
                curriculum.record(bool(stats["satisfied"]))
        else:
            stats = rl_trainer.train_episode(env)

        all_rewards.append(stats["total_reward"])
        all_lengths.append(float(stats["n_selected"]))
        all_satisfied.append(stats["satisfied"])

        if episode % config.PRINT_INTERVAL == 0:
            mean_s = float(np.mean(all_satisfied[-config.PRINT_INTERVAL:]))
            stage_str = (
                f"  stage={curriculum.stage}" if curriculum is not None else ""
            )
            print(
                f"  Ep {episode:>5d} | "
                f"avg_reward={rl_trainer.mean_reward:+.3f} | "
                f"avg_sensors={rl_trainer.mean_length:.2f} | "
                f"success_rate={mean_s:.2f}{stage_str}"
            )

        if episode % config.EVAL_INTERVAL == 0:
            res = evaluate_policy(policy, env, n_episodes=1)
            tag = "+" if res["satisfied"] else "x"
            print(
                f"  -- eval --  {tag}  sensors={res['n_selected']}  "
                f"trace={res['trace']:.4f}"
            )

    # ── 6. Final evaluation ───────────────────────────────────────────────────
    res = evaluate_policy(policy, env, n_episodes=1)
    print(f"\n{'='*60}")
    print("  Final evaluation (deterministic greedy decoding)")
    print(f"  GNN+RL  : {res['n_selected']} sensors  "
          f"(trace={res['trace']:.4f}, satisfied={res['satisfied']})")
    print(f"  Greedy  : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f})")

    # Beam search evaluation (Plan D)
    if args.beam_width > 1:
        beam_res = rl_trainer.beam_rollout(env, n_rollouts=args.beam_width, rng_seed=0)
        print(f"  GNN+RL beam ({args.beam_width} rollouts): "
              f"{beam_res['n_selected']} sensors  "
              f"(trace={beam_res['trace']:.4f}, satisfied={beam_res['satisfied']})")
    print(f"{'='*60}\n")

    # ── 7. Save checkpoint ────────────────────────────────────────────────────
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "args": vars(args),
            "J": torch.tensor(env.J, dtype=torch.float64),
            "epsilon": torch.tensor(env.epsilon, dtype=torch.float64),
            "signed_adj": signed_adj,
            "band_radius": args.band_radius,
            "step_penalty": args.step_penalty,
            "period_hint": args.period_hint,
        },
        args.checkpoint,
    )
    print(f"  Model saved -> {args.checkpoint}")

    # ── 8. Plot training curves ───────────────────────────────────────────────
    if not args.no_plot:
        window = min(50, max(args.episodes // 10, 1))
        smooth = lambda x: np.convolve(x, np.ones(window) / window, mode="valid")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(smooth(all_rewards), color="steelblue")
        axes[0].set_title("Episode Reward (smoothed)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")

        axes[1].plot(smooth(all_lengths), color="coral")
        axes[1].axhline(n_greedy, color="grey", linestyle="--", label="Greedy")
        axes[1].set_title("Sensors Selected (smoothed)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("# Sensors")
        axes[1].legend()

        axes[2].plot(smooth(all_satisfied), color="green")
        axes[2].set_ylim([-0.05, 1.05])
        axes[2].set_title("Success Rate (smoothed)")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Fraction Satisfied")

        plt.tight_layout()
        plt.savefig(config.PLOT_PATH, dpi=120)
        print(f"  Training curve  -> {config.PLOT_PATH}")


if __name__ == "__main__":
    main()



# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN+RL Sensor Selection")
    # Problem
    p.add_argument("--N", type=int, default=config.N)
    p.add_argument("--sigma", type=float, default=config.SIGMA)
    p.add_argument("--eps_frac", type=float, default=config.EPSILON_FRAC)
    p.add_argument("--kernel_type", type=str, default=config.KERNEL_TYPE,
                   choices=["matern", "j0"])
    p.add_argument("--nu", type=float, default=config.KERNEL_NU)
    p.add_argument("--length_scale", type=float, default=config.KERNEL_LENGTH_SCALE)
    # Training
    p.add_argument("--episodes", type=int, default=config.N_TRAIN_EPISODES)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    # GNN architecture
    p.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM)
    p.add_argument("--n_layers", type=int, default=config.N_GNN_LAYERS)
    p.add_argument("--layer_type", type=str, default=config.LAYER_TYPE,
                   choices=["sage", "gat"])
    p.add_argument("--n_heads", type=int, default=config.N_HEADS)
    p.add_argument("--attention_dropout", type=float, default=config.ATTENTION_DROPOUT)
    p.add_argument("--use_residual", action="store_true",
                   default=config.USE_RESIDUAL)
    p.add_argument("--use_attention_pooling", action="store_true",
                   default=config.USE_ATTENTION_POOLING)
    # Scalability enhancements for large N / J₀ kernels
    p.add_argument("--band_radius", type=int, default=config.BAND_RADIUS,
                   help="Sparse adjacency band radius (0=dense, >0=O(N·r) edges)")
    p.add_argument("--step_penalty", type=float, default=config.STEP_PENALTY,
                   help="Per-step sensor cost multiplier (default 1.0; try 2.0)")
    p.add_argument("--beam_width", type=int, default=config.BEAM_WIDTH,
                   help="Stochastic rollouts at eval time (1=greedy, >1=beam)")
    p.add_argument("--period_hint", type=float, default=config.PERIOD_HINT,
                   help="J₀ period hint for Fourier features (0=disabled; "
                        "recommended: 2.4 * length_scale)")
    # Imitation warm-start (Idea 1)
    p.add_argument("--imitation_episodes", type=int,
                   default=config.IMITATION_EPISODES)
    p.add_argument("--imitation_decay", type=int,
                   default=config.IMITATION_DECAY_EPISODES)
    # Multi-environment / curriculum (Idea 4)
    p.add_argument("--multi_env", action="store_true", default=config.MULTI_ENV)
    p.add_argument("--n_train_envs", type=int, default=config.N_TRAIN_ENVS)
    p.add_argument("--n_min", type=int, default=config.N_MIN)
    p.add_argument("--n_max", type=int, default=config.N_MAX)
    p.add_argument("--curriculum", action="store_true", default=config.CURRICULUM)
    # Misc
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def build_primary_env(args: argparse.Namespace) -> SensorSelectionEnv:
    """Build the primary (single) training environment from CLI args."""
    MAT_FILE = "Kernal_-30.mat"
    MAT_VAR = "Kernal_exp"
    if os.path.exists(MAT_FILE) and args.kernel_type == "matern":
        mat = sio.loadmat(MAT_FILE)
        J_full = mat[MAT_VAR]
        J = np.array(J_full[:args.N, :args.N], dtype=float)
    else:
        J = build_toeplitz_matrix(
            args.N, nu=args.nu, length_scale=args.length_scale,
            kernel_type=args.kernel_type
        )
    epsilon = args.eps_frac * float(np.trace(J))
    return SensorSelectionEnv(
        J, sigma=args.sigma, epsilon=epsilon,
        band_radius=args.band_radius,
        step_penalty=args.step_penalty,
        period_hint=args.period_hint,
    )


def greedy_baseline(env: SensorSelectionEnv) -> tuple[int, float]:
    """Greedy oracle returning (n_selected, final_trace)."""
    traj, trace = get_greedy_trajectory(env)
    return len(traj), trace


def evaluate_policy(
    policy: GNNPolicy,
    env: SensorSelectionEnv,
    n_episodes: int = 1,
) -> dict:
    """Evaluate a trained policy deterministically."""
    policy.eval()
    results = []
    adj_neg_tensor = (
        torch.tensor(env.adj_neg, dtype=torch.float32)
        if policy.signed_adj
        else None
    )
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            node_features, adj, _, selected = state
            adj_tensor = torch.tensor(adj, dtype=torch.float32)
            done = False
            while not done:
                nf = torch.tensor(node_features, dtype=torch.float32)
                mask = torch.tensor(~selected, dtype=torch.bool)
                if not mask.any():
                    break
                action, _, _, _ = policy.get_action(
                    nf, adj_tensor, mask, deterministic=True,
                    adj_neg=adj_neg_tensor
                )
                state, _, done, _ = env.step(action)
                node_features, _, _, selected = state
            results.append({
                "n_selected": env.n_selected,
                "trace": env.current_trace,
                "satisfied": env.is_satisfied,
            })
    policy.train()
    return results[0]


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Automatically enable signed_adj for J0 kernel
    signed_adj = config.SIGNED_ADJ or (args.kernel_type == "j0")

    print(f"\n{'='*60}")
    print("  GNN + REINFORCE  -  Submodular Cover Sensor Selection")
    print(f"{'='*60}")
    print(f"  N={args.N}  sigma={args.sigma}  eps_frac={args.eps_frac}")
    print(f"  kernel={args.kernel_type}  nu={args.nu}  l={args.length_scale}")
    print(f"  layer={args.layer_type}  residual={args.use_residual}  "
          f"attn_pool={args.use_attention_pooling}  signed_adj={signed_adj}")

    # ── 1. Build primary environment ─────────────────────────────────────────
    env = build_primary_env(args)
    print(f"\n  trace(J) = {np.trace(env.J):.4f}   "
          f"epsilon = {env.epsilon:.4f}  j_is_pd={env.j_is_pd}")

    # ── 2. Greedy baseline & trajectory ──────────────────────────────────────
    greedy_traj, trace_greedy = get_greedy_trajectory(env)
    n_greedy = len(greedy_traj)
    print(f"\n  Greedy baseline : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f}, satisfied={trace_greedy <= env.epsilon})")

    # ── 3. Build model ────────────────────────────────────────────────────────
    policy = GNNPolicy(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        layer_type=args.layer_type,
        n_heads=args.n_heads,
        attention_dropout=args.attention_dropout,
        use_residual=args.use_residual,
        use_attention_pooling=args.use_attention_pooling,
        signed_adj=signed_adj,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  GNN parameters  : {n_params:,}")

    rl_trainer = REINFORCETrainer(
        policy,
        lr=args.lr,
        gamma=config.GAMMA,
        entropy_coef=config.ENTROPY_COEF,
        value_loss_coef=config.VALUE_LOSS_COEF,
    )

    # ── 4. Imitation pre-training (Idea 1) ────────────────────────────────────
    imitation_losses: list[float] = []
    if args.imitation_episodes > 0 and greedy_traj:
        print(f"\n  Imitation warm-start for {args.imitation_episodes} episodes ...")
        imitator = ImitationTrainer(policy, lr=args.lr)
        for ep in range(1, args.imitation_episodes + 1):
            stats = imitator.train_episode(env, greedy_traj)
            imitation_losses.append(stats["imitation_loss"])
            if ep % config.PRINT_INTERVAL == 0:
                print(f"  Imitation ep {ep:>5d} | "
                      f"avg_loss={imitator.mean_loss:.4f}")

    # ── 5. RL training (with optional blended annealing) ──────────────────────
    print(f"\n  RL training for {args.episodes} episodes ...")

    # Multi-env setup (Idea 4)
    if args.multi_env:
        if args.curriculum:
            curriculum = CurriculumScheduler(
                n_min=args.n_min, n_max=args.n_max, seed=args.seed
            )
            sample_env = curriculum.sample
        else:
            generator = ProblemInstanceGenerator(
                n_range=(args.n_min, args.n_max),
                kernel_types=[args.kernel_type],
                seed=args.seed,
            )
            curriculum = None
            sample_env = generator.sample
        # Pre-generate a pool of environments
        train_envs = [sample_env() for _ in range(args.n_train_envs)]
        train_trajs = [get_greedy_trajectory(e)[0] for e in train_envs]
    else:
        train_envs = [env]
        train_trajs = [greedy_traj]
        curriculum = None

    all_rewards: list[float] = []
    all_lengths: list[float] = []
    all_satisfied: list[float] = []

    for episode in range(1, args.episodes + 1):
        # Compute blended imitation coefficient (anneals 1 -> 0)
        if args.imitation_decay > 0 and episode <= args.imitation_decay:
            imitation_coef = config.IMITATION_COEF * (
                1.0 - (episode - 1) / args.imitation_decay
            )
        else:
            imitation_coef = 0.0

        if args.multi_env:
            stats = rl_trainer.train_multi_env_episode(
                train_envs,
                greedy_trajectories=train_trajs,
                imitation_coef=imitation_coef,
            )
            if curriculum is not None:
                curriculum.record(bool(stats["satisfied"]))
        else:
            stats = rl_trainer.train_episode(
                env,
                greedy_trajectory=greedy_traj,
                imitation_coef=imitation_coef,
            )

        all_rewards.append(stats["total_reward"])
        all_lengths.append(float(stats["n_selected"]))
        all_satisfied.append(stats["satisfied"])

        if episode % config.PRINT_INTERVAL == 0:
            mean_s = float(np.mean(all_satisfied[-config.PRINT_INTERVAL:]))
            stage_str = (
                f"  stage={curriculum.stage}" if curriculum is not None else ""
            )
            print(
                f"  Ep {episode:>5d} | "
                f"avg_reward={rl_trainer.mean_reward:+.3f} | "
                f"avg_sensors={rl_trainer.mean_length:.2f} | "
                f"success_rate={mean_s:.2f} | "
                f"imit_coef={imitation_coef:.3f}{stage_str}"
            )

        if episode % config.EVAL_INTERVAL == 0:
            res = evaluate_policy(policy, env, n_episodes=1)
            tag = "+" if res["satisfied"] else "x"
            print(
                f"  -- eval --  {tag}  sensors={res['n_selected']}  "
                f"trace={res['trace']:.4f}"
            )

    # ── 6. Final evaluation ───────────────────────────────────────────────────
    res = evaluate_policy(policy, env, n_episodes=1)
    print(f"\n{'='*60}")
    print("  Final evaluation (deterministic greedy decoding)")
    print(f"  GNN+RL  : {res['n_selected']} sensors  "
          f"(trace={res['trace']:.4f}, satisfied={res['satisfied']})")
    print(f"  Greedy  : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f})")

    # Beam search evaluation (Plan D)
    if args.beam_width > 1:
        beam_res = rl_trainer.beam_rollout(env, n_rollouts=args.beam_width, rng_seed=0)
        print(f"  GNN+RL beam ({args.beam_width} rollouts): "
              f"{beam_res['n_selected']} sensors  "
              f"(trace={beam_res['trace']:.4f}, satisfied={beam_res['satisfied']})")
    print(f"{'='*60}\n")

    # ── 7. Save checkpoint ────────────────────────────────────────────────────
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "args": vars(args),
            "J": torch.tensor(env.J, dtype=torch.float64),
            "epsilon": torch.tensor(env.epsilon, dtype=torch.float64),
            "greedy_trajectory": greedy_traj,
            "signed_adj": signed_adj,
            "band_radius": args.band_radius,
            "step_penalty": args.step_penalty,
            "period_hint": args.period_hint,
        },
        args.checkpoint,
    )
    print(f"  Model saved -> {args.checkpoint}")

    # ── 8. Plot training curves ───────────────────────────────────────────────
    if not args.no_plot:
        window = min(50, max(args.episodes // 10, 1))
        smooth = lambda x: np.convolve(x, np.ones(window) / window, mode="valid")

        n_plots = 4 if imitation_losses else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

        idx = 0
        if imitation_losses:
            axes[idx].plot(imitation_losses, color="purple")
            axes[idx].set_title("Imitation Loss")
            axes[idx].set_xlabel("Episode")
            idx += 1

        axes[idx].plot(smooth(all_rewards), color="steelblue")
        axes[idx].set_title("Episode Reward (smoothed)")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Reward")
        idx += 1

        axes[idx].plot(smooth(all_lengths), color="coral")
        axes[idx].axhline(n_greedy, color="grey", linestyle="--", label="Greedy")
        axes[idx].set_title("Sensors Selected (smoothed)")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("# Sensors")
        axes[idx].legend()
        idx += 1

        axes[idx].plot(smooth(all_satisfied), color="green")
        axes[idx].set_ylim([-0.05, 1.05])
        axes[idx].set_title("Success Rate (smoothed)")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Fraction Satisfied")

        plt.tight_layout()
        plt.savefig(config.PLOT_PATH, dpi=120)
        print(f"  Training curve  -> {config.PLOT_PATH}")


if __name__ == "__main__":
    main()
