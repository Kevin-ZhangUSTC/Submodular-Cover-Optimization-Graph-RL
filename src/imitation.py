"""
Imitation learning (behavioural cloning) warm-start for the GNN+RL policy.

This module provides:

1. ``get_greedy_trajectory(env)``
   Run the greedy maximum-marginal-gain oracle and return the ordered sequence
   of node indices it selects, together with the final trace value.

2. ``ImitationTrainer``
   Pre-train a GNNPolicy to imitate the greedy oracle via behavioural cloning
   (cross-entropy loss).  After warm-start, RL fine-tuning takes over.

The blended training mode (``imitation_coef > 0`` during RL) is handled
inside ``REINFORCETrainer`` (see ``rl_trainer.py``); this module only
handles the pure imitation phase.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from .environment import SensorSelectionEnv
from .gnn_model import GNNPolicy


# ──────────────────────────────────────────────────────────────────────────────
# Greedy oracle trajectory
# ──────────────────────────────────────────────────────────────────────────────

def get_greedy_trajectory(env: SensorSelectionEnv) -> Tuple[List[int], float]:
    """Run the greedy maximum-marginal-gain oracle and record the trajectory.

    At each step the sensor with the greatest reduction in posterior trace is
    selected.  The episode ends as soon as the trace constraint is satisfied.

    Parameters
    ----------
    env : SensorSelectionEnv
        The environment to solve.  The environment state is **not** modified;
        the function operates on a temporary copy of the selection state.

    Returns
    -------
    trajectory : List[int]
        Ordered sequence of selected node indices.
    final_trace : float
        Posterior trace after applying the greedy selection.
    """
    selected = np.zeros(env.N, dtype=bool)
    current_trace = env.trace_J
    trajectory: List[int] = []

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
        trajectory.append(best_idx)

    return trajectory, current_trace


# ──────────────────────────────────────────────────────────────────────────────
# Imitation trainer
# ──────────────────────────────────────────────────────────────────────────────

class ImitationTrainer:
    """Pre-train a GNNPolicy to imitate the greedy oracle (behavioural cloning).

    Loss per episode:
        L_imitation = -sum_t log pi(a_greedy_t | s_t)

    The trainer accepts an optional *imitation_coef* schedule so that the same
    object can be reused during a blended RL phase where the coefficient is
    annealed to zero.

    Parameters
    ----------
    policy         : GNNPolicy to train
    lr             : Adam learning rate
    grad_clip      : maximum gradient norm
    """

    def __init__(
        self,
        policy: GNNPolicy,
        lr: float = 3e-4,
        grad_clip: float = 1.0,
    ) -> None:
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.grad_clip = grad_clip

        # Running statistics for logging
        self.episode_losses: deque = deque(maxlen=100)

    # --------------------------------------------------------------------------

    def train_episode(
        self,
        env: SensorSelectionEnv,
        trajectory: List[int],
    ) -> dict:
        """Run one behavioural-cloning episode.

        Rolls out the environment following *trajectory* step-by-step,
        computes the cross-entropy loss against the greedy actions, and
        performs one gradient update.

        Parameters
        ----------
        env        : SensorSelectionEnv  (will be reset internally)
        trajectory : List[int]  ordered greedy actions (from get_greedy_trajectory)

        Returns
        -------
        dict with keys: imitation_loss, n_steps
        """
        state = env.reset()
        node_features, adj, _, selected = state
        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        adj_neg_tensor = (
            torch.tensor(env.adj_neg, dtype=torch.float32)
            if self.policy.signed_adj
            else None
        )

        total_loss = torch.tensor(0.0)
        n_steps = 0

        for action in trajectory:
            if selected[action]:
                # Greedy oracle should never revisit, but be defensive
                break

            nf = torch.tensor(node_features, dtype=torch.float32)
            mask = torch.tensor(~selected, dtype=torch.bool)

            if not mask.any():
                break

            logits, _ = self.policy.forward(nf, adj_tensor, mask, adj_neg_tensor)
            # Cross-entropy: -log softmax(logits)[action]
            log_probs = F.log_softmax(logits, dim=0)
            loss = -log_probs[action]
            total_loss = total_loss + loss
            n_steps += 1

            state, _, done, _ = env.step(action)
            node_features, _, _, selected = state
            if done:
                break

        if n_steps > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()

        loss_val = float(total_loss.item()) / max(n_steps, 1)
        self.episode_losses.append(loss_val)
        return {"imitation_loss": loss_val, "n_steps": n_steps}

    # --------------------------------------------------------------------------

    @property
    def mean_loss(self) -> float:
        if len(self.episode_losses) == 0:
            return 0.0
        return float(np.mean(self.episode_losses))
