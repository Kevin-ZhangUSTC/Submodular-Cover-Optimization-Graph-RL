"""
Evaluation utilities for the trained GNN+RL policy.

Compares:
  - GNN+RL policy (greedy deterministic decoding)
  - Greedy oracle (maximum marginal gain at each step)
  - Random baseline (uniformly random sensor selection)
  - Brute-force optimal (exact, only feasible for small N <= 15)

Usage
-----
    # Evaluate on the training problem instance
    python evaluate.py [--checkpoint checkpoint.pt] [--brute-force]

    # Evaluate generalisation on randomly generated held-out instances
    python evaluate.py --eval_multi [--n_eval 30] [--n_min 5] [--n_max 20]

All results are printed as a summary table.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import List

import numpy as np
import torch

import config
from src.kernel import build_toeplitz_matrix
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy
from src.optimal_solver import get_greedy_trajectory
from src.dataset import ProblemInstanceGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────

def run_policy(policy: GNNPolicy, env: SensorSelectionEnv) -> dict:
    """Run the trained policy deterministically and return result stats."""
    policy.eval()
    adj_neg_tensor = (
        torch.tensor(env.adj_neg, dtype=torch.float32)
        if policy.signed_adj
        else None
    )
    with torch.no_grad():
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
                nf, adj_tensor, mask, deterministic=True, adj_neg=adj_neg_tensor
            )
            state, _, done, _ = env.step(action)
            node_features, _, _, selected = state
    return {
        "n_selected": env.n_selected,
        "trace": env.current_trace,
        "satisfied": env.is_satisfied,
        "selected_mask": env.selected.copy(),
    }


def run_greedy(env: SensorSelectionEnv) -> dict:
    """Greedy maximum-marginal-gain baseline."""
    traj, trace = get_greedy_trajectory(env)
    selected = np.zeros(env.N, dtype=bool)
    for idx in traj:
        selected[idx] = True
    return {
        "n_selected": int(selected.sum()),
        "trace": trace,
        "satisfied": trace <= env.epsilon,
        "selected_mask": selected,
    }


def run_random(env: SensorSelectionEnv, seed: int = 0) -> dict:
    """Random selection baseline (averaged over multiple seeds)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(env.N)
    state = env.reset()
    _, _, _, selected = state
    done = False
    for action in order:
        if done:
            break
        if not selected[action]:
            state, _, done, _ = env.step(action)
            _, _, _, selected = state
    return {
        "n_selected": env.n_selected,
        "trace": env.current_trace,
        "satisfied": env.is_satisfied,
        "selected_mask": env.selected.copy(),
    }


def run_brute_force(env: SensorSelectionEnv) -> dict:
    """Exact brute-force search (only feasible for N <= 15)."""
    if env.N > 15:
        raise ValueError("Brute-force is only feasible for N <= 15.")

    best_k = env.N
    best_mask = np.ones(env.N, dtype=bool)
    best_trace = 0.0

    for k in range(1, env.N + 1):
        found = False
        for combo in itertools.combinations(range(env.N), k):
            S = np.array(combo)
            t = env._compute_posterior_trace(S)
            if t <= env.epsilon:
                best_k = k
                best_mask = np.zeros(env.N, dtype=bool)
                best_mask[S] = True
                best_trace = t
                found = True
                break
        if found:
            break

    return {
        "n_selected": best_k,
        "trace": best_trace,
        "satisfied": best_trace <= env.epsilon,
        "selected_mask": best_mask,
    }


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GNN+RL sensor selection")
    p.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    p.add_argument("--brute-force", action="store_true",
                   help="Run brute-force (only for small N)")
    p.add_argument("--n_random", type=int, default=10,
                   help="Number of random-baseline runs to average")
    p.add_argument("--beam_width", type=int, default=config.BEAM_WIDTH,
                   help="Stochastic rollouts for beam search (1=greedy, >1=beam)")
    # Multi-instance evaluation (Idea 4)
    p.add_argument("--eval_multi", action="store_true",
                   help="Evaluate generalisation on randomly generated instances")
    p.add_argument("--n_eval", type=int, default=30,
                   help="Number of held-out instances for eval_multi")
    p.add_argument("--n_min", type=int, default=config.N_MIN)
    p.add_argument("--n_max", type=int, default=config.N_MAX)
    p.add_argument("--kernel_type", type=str, default=None,
                   help="Kernel type for generated instances (default: from checkpoint)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def _load_policy_and_env(args: argparse.Namespace):
    """Load checkpoint and reconstruct policy + primary env."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    J = ckpt["J"].numpy()
    epsilon = float(ckpt["epsilon"])
    signed_adj = ckpt.get("signed_adj", saved_args.get("signed_adj", False))
    band_radius = ckpt.get("band_radius", saved_args.get("band_radius", 0))
    step_penalty = ckpt.get("step_penalty", saved_args.get("step_penalty", 1.0))
    period_hint = ckpt.get("period_hint", saved_args.get("period_hint", 0.0))

    env = SensorSelectionEnv(
        J, sigma=saved_args["sigma"], epsilon=epsilon,
        band_radius=band_radius,
        step_penalty=step_penalty,
        period_hint=period_hint,
    )

    policy = GNNPolicy(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=saved_args.get("hidden_dim", config.HIDDEN_DIM),
        n_layers=saved_args.get("n_layers", config.N_GNN_LAYERS),
        layer_type=saved_args.get("layer_type", config.LAYER_TYPE),
        n_heads=saved_args.get("n_heads", config.N_HEADS),
        attention_dropout=saved_args.get("attention_dropout", config.ATTENTION_DROPOUT),
        use_residual=saved_args.get("use_residual", config.USE_RESIDUAL),
        use_attention_pooling=saved_args.get(
            "use_attention_pooling", config.USE_ATTENTION_POOLING
        ),
        signed_adj=signed_adj,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, env, ckpt


def main() -> None:
    args = parse_args()

    policy, env, ckpt = _load_policy_and_env(args)
    saved_args = ckpt["args"]
    N = env.N

    print(f"\n{'='*60}")
    print("  Evaluation Report")
    print(f"{'='*60}")
    print(f"  N={N}  sigma={saved_args['sigma']:.3f}  "
          f"trace(J)={np.trace(env.J):.4f}  epsilon={env.epsilon:.4f}\n")

    # ── Standard single-instance evaluation ──────────────────────────────────
    from src.rl_trainer import REINFORCETrainer

    t0 = time.perf_counter()
    rl_res = run_policy(policy, env)
    dt_rl = time.perf_counter() - t0

    t0 = time.perf_counter()
    gr_res = run_greedy(env)
    dt_gr = time.perf_counter() - t0

    rand_ns = []
    for s in range(args.n_random):
        r = run_random(env, seed=s)
        if r["satisfied"]:
            rand_ns.append(r["n_selected"])
    rand_mean = float(np.mean(rand_ns)) if rand_ns else float("inf")

    bf_res = None
    if args.brute_force:
        if N > 15:
            print("  (brute-force skipped: N > 15)")
        else:
            t0 = time.perf_counter()
            bf_res = run_brute_force(env)
            dt_bf = time.perf_counter() - t0

    # Beam search evaluation (Plan D)
    beam_res = None
    dt_beam = 0.0
    if args.beam_width > 1:
        # Create a temporary trainer (no training, just uses beam_rollout)
        tmp_trainer = REINFORCETrainer(policy, lr=0.0)
        t0 = time.perf_counter()
        beam_res = tmp_trainer.beam_rollout(env, n_rollouts=args.beam_width, rng_seed=0)
        dt_beam = time.perf_counter() - t0

    header = f"  {'Method':<25} {'# Sensors':>10} {'Trace':>12} {'Satisfied':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    tag = lambda s: "Yes" if s else "No"
    print(f"  {'GNN+RL (greedy decode)':<25} {rl_res['n_selected']:>10d} "
          f"{rl_res['trace']:>12.4f} {tag(rl_res['satisfied']):>10}")
    if beam_res is not None:
        label = f"GNN+RL (beam k={args.beam_width})"
        print(f"  {label:<25} {beam_res['n_selected']:>10d} "
              f"{beam_res['trace']:>12.4f} {tag(beam_res['satisfied']):>10}")
    print(f"  {'Greedy':<25} {gr_res['n_selected']:>10d} "
          f"{gr_res['trace']:>12.4f} {tag(gr_res['satisfied']):>10}")
    print(f"  {'Random (avg)':<25} {rand_mean:>10.1f} "
          f"{'-':>12} {'-':>10}")
    if bf_res is not None:
        print(f"  {'Brute-force (opt.)':<25} {bf_res['n_selected']:>10d} "
              f"{bf_res['trace']:>12.4f} {tag(bf_res['satisfied']):>10}")

    print(f"\n  GNN+RL solve time   : {dt_rl*1e3:.2f} ms")
    if beam_res is not None:
        print(f"  Beam search time    : {dt_beam*1e3:.2f} ms  "
              f"(k={args.beam_width} rollouts)")
    print(f"  Greedy solve time   : {dt_gr*1e3:.2f} ms")
    if bf_res is not None:
        print(f"  Brute-force time    : {dt_bf*1e3:.2f} ms")

    # ── Multi-instance generalisation evaluation (Idea 4) ────────────────────
    if args.eval_multi:
        print(f"\n{'='*60}")
        print(f"  Generalisation Evaluation  ({args.n_eval} held-out instances)")
        print(f"{'='*60}")
        kernel_type = args.kernel_type or saved_args.get("kernel_type", "matern")
        gen = ProblemInstanceGenerator(
            n_range=(args.n_min, args.n_max),
            kernel_types=[kernel_type],
            seed=99,
        )
        rl_ns, gr_ns, rl_sat, gr_sat = [], [], [], []
        for _ in range(args.n_eval):
            test_env = gen.sample()
            rl_r = run_policy(policy, test_env)
            gr_r = run_greedy(test_env)
            rl_ns.append(rl_r["n_selected"])
            gr_ns.append(gr_r["n_selected"])
            rl_sat.append(float(rl_r["satisfied"]))
            gr_sat.append(float(gr_r["satisfied"]))

        print(f"  {'Method':<20} {'Avg # Sensors':>15} {'Success Rate':>14}")
        print("  " + "-" * 52)
        print(f"  {'GNN+RL':<20} {np.mean(rl_ns):>15.2f} {np.mean(rl_sat):>14.2f}")
        print(f"  {'Greedy':<20} {np.mean(gr_ns):>15.2f} {np.mean(gr_sat):>14.2f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
