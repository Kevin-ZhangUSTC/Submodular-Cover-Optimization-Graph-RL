"""
run_benchmark.py — Compare GNN+RL (with imitation warm-start) against the
greedy oracle across multiple Matérn kernel problem instances.

Usage:
    python run_benchmark.py [--episodes 500] [--seeds 3] [--no-plot]

Results are printed to stdout and saved to benchmark_results.txt.
"""

from __future__ import annotations
import argparse, time
import numpy as np
import torch

from src.kernel import build_toeplitz_matrix
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy
from src.rl_trainer import REINFORCETrainer
from src.imitation import ImitationTrainer, get_greedy_trajectory
from src.dataset import ProblemInstanceGenerator


INSTANCES = [
    dict(N=10, nu=1.5, ls=3.0, sigma=0.5, eps_frac=0.25, label="N=10 ν=1.5 l=3"),
    dict(N=10, nu=0.5, ls=3.0, sigma=0.5, eps_frac=0.25, label="N=10 ν=0.5 l=3"),
    dict(N=10, nu=2.5, ls=2.0, sigma=0.4, eps_frac=0.20, label="N=10 ν=2.5 l=2"),
    dict(N=12, nu=1.5, ls=4.0, sigma=0.8, eps_frac=0.30, label="N=12 ν=1.5 l=4"),
    dict(N=15, nu=1.5, ls=2.0, sigma=0.5, eps_frac=0.20, label="N=15 ν=1.5 l=2"),
    dict(N=15, nu=0.5, ls=3.0, sigma=0.3, eps_frac=0.20, label="N=15 ν=0.5 l=3"),
    dict(N=20, nu=1.5, ls=3.0, sigma=0.5, eps_frac=0.22, label="N=20 ν=1.5 l=3"),
    dict(N=20, nu=2.5, ls=2.5, sigma=0.6, eps_frac=0.18, label="N=20 ν=2.5 l=2.5"),
]


def make_env(cfg):
    J = build_toeplitz_matrix(cfg["N"], nu=cfg["nu"], length_scale=cfg["ls"])
    return SensorSelectionEnv(J, sigma=cfg["sigma"],
                              epsilon=cfg["eps_frac"] * float(np.trace(J)))


def run_policy(policy, env):
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
    return env.n_selected, env.is_satisfied


def train(env, n_imit=100, n_rl=500, seed=0, hidden=32, n_layers=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    traj, _ = get_greedy_trajectory(env)
    policy = GNNPolicy(node_feat_dim=5, hidden_dim=hidden,
                       n_layers=n_layers, use_residual=True)
    imitator = ImitationTrainer(policy, lr=3e-3)
    for _ in range(n_imit):
        imitator.train_episode(env, traj)
    trainer = REINFORCETrainer(policy, lr=3e-3, entropy_coef=0.05)
    for ep in range(n_rl):
        coef = 0.3 * max(0.0, 1.0 - ep / 200.0)
        trainer.train_episode(env, greedy_trajectory=traj, imitation_coef=coef)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--output", type=str, default="benchmark_results.txt")
    args = ap.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("=" * 74)
    p("  GNN+RL (Imitation Warm-Start) vs Greedy Oracle")
    p(f"  RL episodes: {args.episodes} | Imitation: {args.episodes//5} | Seeds: {args.seeds}")
    p("=" * 74)
    p(f"  {'Instance':<22} {'Greedy':>7} {'RL best':>8} {'RL mean':>8} {'Succ':>7} {'Δ':>5}")
    p("  " + "-" * 60)

    rows = []
    for cfg in INSTANCES:
        env = make_env(cfg)
        traj, _ = get_greedy_trajectory(env)
        n_gr = len(traj)
        ns, sats = [], []
        for seed in range(args.seeds):
            pol = train(env, n_imit=args.episodes//5, n_rl=args.episodes,
                        seed=seed, hidden=args.hidden, n_layers=args.n_layers)
            n, s = run_policy(pol, env)
            ns.append(n); sats.append(int(s))
        sat_ns = [n for n, s in zip(ns, sats) if s]
        best = min(sat_ns) if sat_ns else min(ns)
        delta = n_gr - best
        sym = "✓" if delta > 0 else "="
        p(f"  {cfg['label']:<22} {n_gr:>7d} {best:>8d} {np.mean(ns):>8.1f} "
          f"{np.mean(sats):>7.0%} {sym}{delta:>+4d}")
        rows.append(dict(n_gr=n_gr, best=best, succ=np.mean(sats), delta=delta))

    p()
    p(f"  100% success  : {sum(1 for r in rows if r['succ']==1.0)}/{len(rows)} instances")
    p(f"  Beats greedy  : {sum(1 for r in rows if r['delta']>0)}/{len(rows)} instances")
    p(f"  Ties  greedy  : {sum(1 for r in rows if r['delta']==0)}/{len(rows)} instances")
    p(f"  Avg Δsensors  : {np.mean([r['delta'] for r in rows]):+.2f}")
    p("=" * 74)

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
