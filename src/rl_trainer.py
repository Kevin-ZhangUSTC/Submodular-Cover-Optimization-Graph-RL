"""
REINFORCE with Actor-Critic baseline trainer.

Algorithm
---------
For each episode:
    1. Roll out one full trajectory using the current GNN policy.
    2. Compute discounted returns G_t.
    3. Update the policy with the REINFORCE loss:
           L_pi = -sum_t log pi(a_t | s_t) * (G_t - V(s_t))
       and the value loss:
           L_V = sum_t (G_t - V(s_t))^2
       plus an entropy bonus for exploration.

Optional blended imitation loss
---------------------------------
When ``imitation_coef > 0`` and a ``greedy_trajectory`` is supplied the
trainer also computes a behavioural-cloning cross-entropy term and adds it
to the policy gradient loss:
    L_total = L_pi + value_loss_coef * L_V + entropy_coef * L_entropy
            + imitation_coef * L_imitation

The blended coefficient can be annealed to zero over training to implement
the curriculum transition from Idea 1.
"""

from __future__ import annotations

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List, Optional, Tuple

from .environment import SensorSelectionEnv
from .gnn_model import GNNPolicy


class REINFORCETrainer:
    """Train a GNNPolicy using REINFORCE with an Actor-Critic baseline.

    Parameters
    ----------
    policy          : GNNPolicy to train
    lr              : Adam learning-rate
    gamma           : discount factor
    entropy_coef    : entropy bonus coefficient
    value_loss_coef : coefficient for the value-function MSE loss
    grad_clip       : maximum gradient norm
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
        greedy_trajectory: Optional[List[int]] = None,
        imitation_coef: float = 0.0,
    ) -> Dict[str, float]:
        """Run one episode and perform a policy-gradient update.

        Parameters
        ----------
        env                : the environment to train on
        greedy_trajectory  : optional greedy action sequence for blended loss
        imitation_coef     : weight of behavioural-cloning term (0 = pure RL)

        Returns
        -------
        dict with keys: total_reward, n_selected, trace, satisfied, policy_loss,
                        value_loss, entropy, imitation_loss
        """
        state = env.reset()
        node_features, adj, _, selected = state
        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        adj_neg_tensor = (
            torch.tensor(env.adj_neg, dtype=torch.float32)
            if self.policy.signed_adj
            else None
        )

        # -- trajectory buffers ------------------------------------------------
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
                nf, adj_tensor, mask, adj_neg=adj_neg_tensor
            )

            state, reward, done, info = env.step(action)
            node_features, _, _, selected = state

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

        # -- compute discounted returns ----------------------------------------
        returns = self._compute_returns(rewards)

        # -- compute losses ----------------------------------------------------
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)

        for log_prob, value, R, ent in zip(log_probs, values, returns, entropies):
            advantage = R - value.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + F.mse_loss(value.unsqueeze(0), R.unsqueeze(0))
            entropy_loss = entropy_loss - ent

        # -- optional blended imitation loss -----------------------------------
        imitation_loss = torch.tensor(0.0)
        if imitation_coef > 0.0 and greedy_trajectory:
            imitation_loss = self._compute_imitation_loss(
                env, greedy_trajectory, adj_tensor, adj_neg_tensor
            )

        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
            + imitation_coef * imitation_loss
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
            "imitation_loss": float(imitation_loss.item()),
        }

    # ──────────────────────────────────────────────────────────────────────────

    def train_multi_env_episode(
        self,
        envs: List[SensorSelectionEnv],
        greedy_trajectories: Optional[List[Optional[List[int]]]] = None,
        imitation_coef: float = 0.0,
    ) -> Dict[str, float]:
        """Sample a random environment from *envs* and run one training episode.

        Parameters
        ----------
        envs                 : pool of environments to train on
        greedy_trajectories  : optional per-env greedy trajectories (same order)
        imitation_coef       : blended imitation coefficient

        Returns
        -------
        Same dict as ``train_episode``, with an extra key ``env_index``.
        """
        idx = random.randrange(len(envs))
        env = envs[idx]
        traj = (
            greedy_trajectories[idx]
            if greedy_trajectories is not None
            else None
        )
        stats = self.train_episode(env, greedy_trajectory=traj,
                                   imitation_coef=imitation_coef)
        stats["env_index"] = float(idx)
        return stats

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_imitation_loss(
        self,
        env: SensorSelectionEnv,
        trajectory: List[int],
        adj_tensor: torch.Tensor,
        adj_neg_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute behavioural-cloning cross-entropy loss against *trajectory*.

        The environment is reset and replayed along the greedy trajectory; at
        each step the cross-entropy of the policy w.r.t. the greedy action is
        accumulated.
        """
        state = env.reset()
        node_features, _, _, selected = state
        total_loss = torch.tensor(0.0)
        n_steps = 0

        for action in trajectory:
            if selected[action]:
                break
            nf = torch.tensor(node_features, dtype=torch.float32)
            mask = torch.tensor(~selected, dtype=torch.bool)
            if not mask.any():
                break
            logits, _ = self.policy.forward(nf, adj_tensor, mask, adj_neg_tensor)
            log_probs = F.log_softmax(logits, dim=0)
            total_loss = total_loss - log_probs[action]
            n_steps += 1
            state, _, done, _ = env.step(action)
            node_features, _, _, selected = state
            if done:
                break

        return total_loss / max(n_steps, 1)

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute normalised discounted returns G_t = sum_{k>=t} gamma^{k-t} r_k."""
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
