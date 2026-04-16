"""
Mini-batch REINFORCE with Actor-Critic baseline trainer.

Algorithm
---------
For each update step:
    1. Roll out *batch_size* full trajectories using the current GNN policy.
    2. Compute discounted returns G_t for each trajectory.
    3. Average the per-trajectory losses and update with a single gradient step:
           L_pi = -sum_t log pi(a_t | s_t) * (G_t - V(s_t))
           L_V  =  sum_t (G_t - V(s_t))^2
       plus an entropy bonus for exploration.
    Averaging over batch_size trajectories reduces gradient variance by
    roughly sqrt(batch_size) compared to single-episode REINFORCE.

When batch_size=1 the method degrades to standard single-episode REINFORCE,
preserving full backward-compatibility.

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
        loss, stats = self._rollout_and_loss(env, greedy_trajectory, imitation_coef)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        self.episode_rewards.append(stats["total_reward"])
        self.episode_lengths.append(float(stats["n_selected"]))
        return stats

    # ──────────────────────────────────────────────────────────────────────────

    def train_batch_episode(
        self,
        envs: List[SensorSelectionEnv],
        batch_size: int = 8,
        greedy_trajectories: Optional[List[Optional[List[int]]]] = None,
        imitation_coef: float = 0.0,
    ) -> Dict[str, float]:
        """Collect *batch_size* trajectories and perform a single gradient update.

        Each trajectory is sampled from a randomly chosen environment in *envs*.
        The per-trajectory losses are averaged before the backward pass, which
        reduces gradient variance by ~sqrt(batch_size) compared to single-episode
        REINFORCE.

        Parameters
        ----------
        envs                 : pool of environments to sample from
        batch_size           : number of trajectories to collect per update
        greedy_trajectories  : optional per-env greedy trajectories (same order as envs)
        imitation_coef       : blended imitation coefficient

        Returns
        -------
        dict with the same keys as ``train_episode`` (averaged over the batch),
        plus ``env_index`` (index of the last sampled environment).
        """
        batch_loss = torch.tensor(0.0)
        stats_sum: Dict[str, float] = {}
        last_idx = 0

        for _ in range(batch_size):
            last_idx = random.randrange(len(envs))
            env = envs[last_idx]
            traj = (
                greedy_trajectories[last_idx]
                if greedy_trajectories is not None
                else None
            )
            episode_loss, stats = self._rollout_and_loss(env, traj, imitation_coef)
            batch_loss = batch_loss + episode_loss / batch_size
            for k, v in stats.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + v / batch_size

        self.optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        self.episode_rewards.append(stats_sum["total_reward"])
        self.episode_lengths.append(stats_sum["n_selected"])
        stats_sum["env_index"] = float(last_idx)
        return stats_sum

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

    def _rollout_and_loss(
        self,
        env: SensorSelectionEnv,
        greedy_trajectory: Optional[List[int]] = None,
        imitation_coef: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Roll out one episode and compute the combined loss (without updating).

        Returns
        -------
        (total_loss, stats_dict)
            total_loss : torch.Tensor with grad_fn (ready for .backward())
            stats_dict : plain-float metrics for logging
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

        stats = {
            "total_reward": float(sum(rewards)),
            "n_selected": float(env.n_selected),
            "trace": float(env.current_trace),
            "satisfied": float(env.is_satisfied),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float((-entropy_loss).item()),
            "imitation_loss": float(imitation_loss.item()),
        }
        return total_loss, stats

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

    # ──────────────────────────────────────────────────────────────────────────

    def beam_rollout(
        self,
        env: SensorSelectionEnv,
        n_rollouts: int = 10,
        rng_seed: int = 0,
    ) -> Dict[str, object]:
        """Run *n_rollouts* **stochastic** rollouts and return the best result.

        This implements Plan D: beam search / stochastic rollouts at test time.
        Even a mediocre learned policy can beat greedy when multiple stochastic
        rollouts are combined, because each rollout explores a different sensor
        ordering guided by the learned probability distribution.

        "Best" means:
          1. Among rollouts that satisfy the constraint, return the one with
             the fewest sensors selected.
          2. If no rollout satisfies the constraint, return the one with the
             lowest posterior trace (closest to feasibility).

        Parameters
        ----------
        env : SensorSelectionEnv
            The environment to evaluate on.  Its state is **not** modified
            permanently (each rollout calls env.reset() internally).
        n_rollouts : int
            Number of stochastic trajectories to sample.
        rng_seed : int
            Seed for PyTorch random sampling (ensures reproducibility).

        Returns
        -------
        dict with keys:
          - ``n_selected``  : int   -- sensors used in the best rollout
          - ``trace``       : float -- posterior trace of the best rollout
          - ``satisfied``   : bool  -- whether the best rollout satisfies ε
          - ``selected_mask``: np.ndarray -- bool mask of selected sensors
          - ``rollout_n``   : int   -- index of the winning rollout
        """
        torch.manual_seed(rng_seed)
        adj_tensor = torch.tensor(env.adj_norm, dtype=torch.float32)
        adj_neg_tensor = (
            torch.tensor(env.adj_neg, dtype=torch.float32)
            if self.policy.signed_adj
            else None
        )

        best: Dict[str, object] = {}
        best_key = (False, env.N + 1, float("inf"))  # (not_sat, n_selected, trace)

        self.policy.eval()
        with torch.no_grad():
            for rollout_idx in range(n_rollouts):
                state = env.reset()
                node_features, _, _, selected = state
                done = False
                while not done:
                    nf = torch.tensor(node_features, dtype=torch.float32)
                    mask = torch.tensor(~selected, dtype=torch.bool)
                    if not mask.any():
                        break
                    # Stochastic sampling (not deterministic)
                    action, _, _, _ = self.policy.get_action(
                        nf, adj_tensor, mask,
                        deterministic=False,
                        adj_neg=adj_neg_tensor,
                    )
                    state, _, done, _ = env.step(action)
                    node_features, _, _, selected = state

                sat = env.is_satisfied
                n_sel = env.n_selected
                trace = env.current_trace
                # Lower key = better: (satisfied DESC, n_selected ASC, trace ASC)
                key = (not sat, n_sel, trace)
                if not best or key < best_key:
                    best_key = key
                    best = {
                        "n_selected": n_sel,
                        "trace": trace,
                        "satisfied": sat,
                        "selected_mask": env.selected.copy(),
                        "rollout_n": rollout_idx,
                    }
        self.policy.train()
        return best
