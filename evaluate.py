"""
Evaluation utilities for the trained GNN+RL policy.

Compares:
  - GNN+RL policy (greedy deterministic decoding)
  - Greedy oracle (maximum marginal gain at each step)
  - Random baseline (uniformly random sensor selection)
  - Brute-force optimal (exact, only feasible for small N ≤ 15)

Usage
-----
    python evaluate.py [--checkpoint checkpoint.pt] [--brute-force]

All results are printed as a summary table.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch

import config
from src.kernel import build_toeplitz_matrix
from src.environment import SensorSelectionEnv
from src.gnn_model import GNNPolicy


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────

def run_policy(policy: GNNPolicy, env: SensorSelectionEnv) -> dict:
    """Run the trained policy deterministically and return result stats."""
    policy.eval()
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
            action, _, _, _ = policy.get_action(nf, adj_tensor, mask,
                                                 deterministic=True)
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
    selected = np.zeros(env.N, dtype=bool)
    current_trace = env.trace_J

    for _ in range(env.N):
        if current_trace <= env.epsilon:
            break
        best_idx, best_trace = -1, current_trace
        for i in range(env.N):
            if selected[i]:
                continue
            candidate = list(np.where(selected)[0]) + [i]
            t = env._compute_posterior_trace(np.array(candidate))
            if t < best_trace:
                best_trace, best_idx = t, i
        if best_idx == -1:
            break
        selected[best_idx] = True
        current_trace = best_trace

    return {
        "n_selected": int(selected.sum()),
        "trace": current_trace,
        "satisfied": current_trace <= env.epsilon,
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
    """Exact brute-force search (only feasible for N ≤ 15)."""
    if env.N > 15:
        raise ValueError("Brute-force is only feasible for N ≤ 15.")

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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved_args = ckpt["args"]
    J = ckpt["J"].numpy()
    epsilon = float(ckpt["epsilon"])

    env = SensorSelectionEnv(J, sigma=saved_args["sigma"], epsilon=epsilon)

    policy = GNNPolicy(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=saved_args.get("hidden_dim", config.HIDDEN_DIM),
        n_layers=saved_args.get("n_layers", config.N_GNN_LAYERS),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])

    N = env.N
    print(f"\n{'='*60}")
    print("  Evaluation Report")
    print(f"{'='*60}")
    print(f"  N={N}  sigma={saved_args['sigma']:.3f}  "
          f"trace(J)={np.trace(J):.4f}  epsilon={epsilon:.4f}\n")

    # ── GNN+RL ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rl_res = run_policy(policy, env)
    dt_rl = time.perf_counter() - t0

    # ── Greedy ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gr_res = run_greedy(env)
    dt_gr = time.perf_counter() - t0

    # ── Random (average) ─────────────────────────────────────────────────────
    rand_ns = []
    for s in range(args.n_random):
        r = run_random(env, seed=s)
        if r["satisfied"]:
            rand_ns.append(r["n_selected"])
    rand_mean = float(np.mean(rand_ns)) if rand_ns else float("inf")

    # ── Brute-force ───────────────────────────────────────────────────────────
    bf_res = None
    if args.brute_force:
        if N > 15:
            print("  (brute-force skipped: N > 15)")
        else:
            t0 = time.perf_counter()
            bf_res = run_brute_force(env)
            dt_bf = time.perf_counter() - t0

    # ── Report ────────────────────────────────────────────────────────────────
    header = f"  {'Method':<20} {'# Sensors':>10} {'Trace':>12} {'Satisfied':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    tag = lambda s: "Yes" if s else "No"
    print(f"  {'GNN+RL':<20} {rl_res['n_selected']:>10d} "
          f"{rl_res['trace']:>12.4f} {tag(rl_res['satisfied']):>10}")
    print(f"  {'Greedy':<20} {gr_res['n_selected']:>10d} "
          f"{gr_res['trace']:>12.4f} {tag(gr_res['satisfied']):>10}")
    print(f"  {'Random (avg)':<20} {rand_mean:>10.1f} "
          f"{'—':>12} {'—':>10}")
    if bf_res is not None:
        print(f"  {'Brute-force (opt.)':<20} {bf_res['n_selected']:>10d} "
              f"{bf_res['trace']:>12.4f} {tag(bf_res['satisfied']):>10}")

    print(f"\n  GNN+RL solve time : {dt_rl*1e3:.2f} ms")
    print(f"  Greedy solve time : {dt_gr*1e3:.2f} ms")
    if bf_res is not None:
        print(f"  Brute-force time  : {dt_bf*1e3:.2f} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
