"""
REINFORCE with Actor-Critic baseline trainer.

Algorithm
---------
For each episode:
    1. Roll out one full trajectory using the current GNN policy.
    2. Compute discounted returns G_t.
    3. Update the policy with the REINFORCE loss:
           L_π = -∑_t log π(a_t | s_t) · (G_t - V(s_t))
       and the value loss:
           L_V = ∑_t (G_t - V(s_t))²
       plus an entropy bonus for exploration.

The adjacency matrix A is fixed throughout training because J is constant;
only the node features change as sensors are sequentially selected.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple

from .environment import SensorSelectionEnv
from .gnn_model import GNNPolicy


class REINFORCETrainer:
    """Train a GNNPolicy using REINFORCE with an Actor-Critic baseline.

    Parameters
    ----------
    policy         : GNNPolicy to train
    lr             : Adam learning-rate
    gamma          : discount factor
    entropy_coef   : entropy bonus coefficient
    value_loss_coef: coefficient for the value-function MSE loss
    grad_clip      : maximum gradient norm
    """

    def __init__(
        self,
        policy: GNNPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.02,
        value_loss_coef: float = 0.5,
        grad_clip: float = 1.0,
    ) -> None:
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_clip = grad_clip

        # Running statistics for logging
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)

    # ──────────────────────────────────────────────────────────────────────────

    def train_episode(
        self,
        env: SensorSelectionEnv,
    ) -> Dict[str, float]:
        """Run one episode and perform a policy-gradient update.

        Returns
        -------
        dict with keys: total_reward, n_selected, trace, satisfied, policy_loss,
                        value_loss, entropy
        """
        state = env.reset()
        node_features, adj, _, selected = state
        adj_tensor = torch.tensor(adj, dtype=torch.float32)   # fixed throughout

        # ── trajectory buffers ────────────────────────────────────────────────
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []

        done = False
        while not done:
            nf = torch.tensor(node_features, dtype=torch.float32)
            mask = torch.tensor(~selected, dtype=torch.bool)

            if not mask.any():
                break

            action, log_prob, entropy, value = self.policy.get_action(
                nf, adj_tensor, mask
            )

            state, reward, done, info = env.step(action)
            node_features, _, _, selected = state

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

        # ── compute discounted returns ────────────────────────────────────────
        returns = self._compute_returns(rewards)

        # ── compute losses ────────────────────────────────────────────────────
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)

        for log_prob, value, R, ent in zip(log_probs, values, returns, entropies):
            advantage = R - value.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + F.mse_loss(value.unsqueeze(0), R.unsqueeze(0))
            entropy_loss = entropy_loss - ent

        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        total_reward = float(sum(rewards))
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(float(env.n_selected))

        return {
            "total_reward": total_reward,
            "n_selected": env.n_selected,
            "trace": env.current_trace,
            "satisfied": float(env.is_satisfied),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float((-entropy_loss).item()),
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute normalised discounted returns G_t = ∑_{k≥t} γ^{k-t} r_k."""
        G: List[float] = []
        running = 0.0
        for r in reversed(rewards):
            running = r + self.gamma * running
            G.insert(0, running)
        returns = torch.tensor(G, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def mean_reward(self) -> float:
        if len(self.episode_rewards) == 0:
            return 0.0
        return float(np.mean(self.episode_rewards))

    @property
    def mean_length(self) -> float:
        if len(self.episode_lengths) == 0:
            return 0.0
        return float(np.mean(self.episode_lengths))
