"""
Main training script for the GNN+RL submodular cover sensor-selection problem.

Usage
-----
    python main.py [--N 20] [--sigma 0.5] [--eps_frac 0.20]
                   [--episodes 2000] [--lr 3e-4] [--seed 42]
                   [--no-plot]

The script will:
  1. Build the Bessel-kernel Toeplitz matrix J.
  2. Instantiate the RL environment.
  3. Train a GNN policy with REINFORCE + baseline.
  4. Save the trained model to ``checkpoint.pt``.
  5. Plot the training curve to ``training_curve.png``.
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

import config
from src.kernel import build_toeplitz_matrix
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy
from src.rl_trainer import REINFORCETrainer


# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN+RL Sensor Selection")
    p.add_argument("--N", type=int, default=config.N,
                   help="Matrix dimension (number of candidate positions)")
    p.add_argument("--sigma", type=float, default=config.SIGMA,
                   help="Observation noise standard deviation")
    p.add_argument("--eps_frac", type=float, default=config.EPSILON_FRAC,
                   help="Trace constraint as fraction of trace(J)")
    p.add_argument("--nu", type=float, default=config.KERNEL_NU,
                   help="Matérn smoothness parameter")
    p.add_argument("--length_scale", type=float, default=config.KERNEL_LENGTH_SCALE,
                   help="Kernel length-scale")
    p.add_argument("--episodes", type=int, default=config.N_TRAIN_EPISODES,
                   help="Number of training episodes")
    p.add_argument("--lr", type=float, default=config.LR,
                   help="Learning rate")
    p.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM,
                   help="GNN hidden dimension")
    p.add_argument("--n_layers", type=int, default=config.N_GNN_LAYERS,
                   help="Number of GNN layers")
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="Random seed")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving the training-curve plot")
    p.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH,
                   help="Path to save the trained model checkpoint")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def greedy_baseline(env: SensorSelectionEnv) -> tuple[int, float]:
    """Greedy oracle: at each step add the sensor with maximum marginal gain.

    This provides a comparison baseline for the RL policy.

    Returns
    -------
    n_selected : int
        Number of sensors chosen by greedy before constraint is satisfied.
    current_trace : float
        Posterior trace after greedy selection.
    """
    selected = np.zeros(env.N, dtype=bool)
    current_trace = env.trace_J

    for _ in range(env.N):
        if current_trace <= env.epsilon:
            break

        best_idx = -1
        best_trace = current_trace

        for i in range(env.N):
            if selected[i]:
                continue
            candidate = list(np.where(selected)[0]) + [i]
            t = env._compute_posterior_trace(np.array(candidate))
            if t < best_trace:
                best_trace = t
                best_idx = i

        if best_idx == -1:
            break
        selected[best_idx] = True
        current_trace = best_trace

    n = int(selected.sum())
    return n, current_trace


def evaluate_policy(
    policy: GNNPolicy,
    env: SensorSelectionEnv,
    n_episodes: int = 20,
) -> dict:
    """Evaluate a trained policy deterministically over multiple resets.

    (The environment is deterministic given a fixed J, so all episodes are
    identical; this is mainly to confirm reproducibility.)
    """
    policy.eval()
    results = []
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
                action, _, _, _ = policy.get_action(nf, adj_tensor, mask,
                                                     deterministic=True)
                state, _, done, _ = env.step(action)
                node_features, _, _, selected = state
            results.append({
                "n_selected": env.n_selected,
                "trace": env.current_trace,
                "satisfied": env.is_satisfied,
            })
    policy.train()
    return results[0]   # deterministic → all identical


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ── 1. Build problem ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GNN + REINFORCE  ─  Submodular Cover Sensor Selection")
    print(f"{'='*60}")
    print(f"  N={args.N}  sigma={args.sigma}  eps_frac={args.eps_frac}"
          f"  nu={args.nu}  l={args.length_scale}")

    J = build_toeplitz_matrix(args.N, nu=args.nu, length_scale=args.length_scale)
    epsilon = args.eps_frac * float(np.trace(J))

    print(f"  trace(J) = {np.trace(J):.4f}   epsilon = {epsilon:.4f}")

    env = SensorSelectionEnv(J, sigma=args.sigma, epsilon=epsilon)

    # ── 2. Greedy baseline ────────────────────────────────────────────────────
    n_greedy, trace_greedy = greedy_baseline(env)
    print(f"\n  Greedy baseline : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f}, satisfied={trace_greedy <= epsilon})")

    # ── 3. Build model ────────────────────────────────────────────────────────
    policy = GNNPolicy(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  GNN parameters  : {n_params:,}")

    trainer = REINFORCETrainer(
        policy,
        lr=args.lr,
        gamma=config.GAMMA,
        entropy_coef=config.ENTROPY_COEF,
        value_loss_coef=config.VALUE_LOSS_COEF,
    )

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(f"\n  Training for {args.episodes} episodes …\n")

    all_rewards: list[float] = []
    all_lengths: list[float] = []
    all_satisfied: list[float] = []

    for episode in range(1, args.episodes + 1):
        stats = trainer.train_episode(env)
        all_rewards.append(stats["total_reward"])
        all_lengths.append(float(stats["n_selected"]))
        all_satisfied.append(stats["satisfied"])

        if episode % config.PRINT_INTERVAL == 0:
            # trainer.mean_reward / mean_length use the rolling 100-episode deques
            mean_s = float(np.mean(all_satisfied[-config.PRINT_INTERVAL:]))
            print(
                f"  Ep {episode:>5d} | "
                f"avg_reward={trainer.mean_reward:+.3f} | "
                f"avg_sensors={trainer.mean_length:.2f} | "
                f"success_rate={mean_s:.2f}"
            )

        if episode % config.EVAL_INTERVAL == 0:
            res = evaluate_policy(policy, env, n_episodes=1)
            tag = "✓" if res["satisfied"] else "✗"
            print(
                f"  ── eval ──  {tag}  sensors={res['n_selected']}  "
                f"trace={res['trace']:.4f}"
            )

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    res = evaluate_policy(policy, env, n_episodes=1)
    print(f"\n{'='*60}")
    print("  Final evaluation (deterministic)")
    print(f"  GNN+RL  : {res['n_selected']} sensors  "
          f"(trace={res['trace']:.4f}, "
          f"satisfied={res['satisfied']})")
    print(f"  Greedy  : {n_greedy} sensors  "
          f"(trace={trace_greedy:.4f})")
    print(f"{'='*60}\n")

    # ── 6. Save checkpoint ────────────────────────────────────────────────────
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "args": vars(args),
            "J": torch.tensor(J, dtype=torch.float64),
            "epsilon": torch.tensor(epsilon, dtype=torch.float64),
        },
        args.checkpoint,
    )
    print(f"  Model saved → {args.checkpoint}")

    # ── 7. Plot training curve ────────────────────────────────────────────────
    if not args.no_plot:
        window = min(50, args.episodes // 10)
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
        print(f"  Training curve  → {config.PLOT_PATH}")


if __name__ == "__main__":
    main()
