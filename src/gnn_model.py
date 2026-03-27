"""
GNN-based policy and value networks for the sensor-selection RL problem.

Architecture
------------
Input  : node features  X  ∈ ℝ^{N×d}  and normalised adjacency  A  ∈ ℝ^{N×N}
Network:
    1. Linear input projection: X → ℝ^{N×H}
    2. K GraphSAGE-style message-passing layers:
           h_new = LayerNorm( ReLU( Linear([h_self || h_agg]) ) )
       where  h_agg = A · h  (neighbourhood aggregation)
    3. Policy head  : per-node linear → scalar logit   (action selection)
    4. Value head   : mean-pool over valid nodes → scalar  (state value)

The policy outputs a masked softmax over unselected nodes; the value head
estimates V(s) used as the REINFORCE baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE-style message-passing layer.

    h_new = LayerNorm( ReLU( W · [h_self ‖ A·h_neigh] ) )

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (N, in_dim)
        adj : (N, N) row-normalised adjacency matrix (zero diagonal)

        Returns
        -------
        h   : (N, out_dim)
        """
        h_neigh = adj @ x                             # (N, in_dim)
        h_cat = torch.cat([x, h_neigh], dim=-1)       # (N, 2·in_dim)
        h = self.norm(F.relu(self.linear(h_cat)))
        return h


class GNNPolicy(nn.Module):
    """Actor-Critic GNN network for sequential sensor selection.

    Parameters
    ----------
    node_feat_dim : int
        Number of input features per node (default 4).
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of GNN message-passing layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gnn_layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Policy head: maps node embedding → scalar action logit
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value head: maps mean node embedding → scalar state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        node_features : (N, feat_dim)
        adj           : (N, N) normalised adjacency
        mask          : (N,) bool tensor — True where action is valid (not yet selected)

        Returns
        -------
        action_logits : (N,)  — raw logits (−∞ for already-selected nodes)
        state_value   : scalar tensor
        """
        x = self.input_proj(node_features)            # (N, H)

        for layer in self.gnn_layers:
            x = layer(x, adj)                         # (N, H)

        # ── action logits ────────────────────────────────────────────────────
        action_logits = self.policy_head(x).squeeze(-1)   # (N,)
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, float("-inf"))

        # ── state value (mean-pool over valid / unselected nodes) ────────────
        if mask is not None and mask.any():
            pooled = x[mask].mean(dim=0)              # (H,)
        else:
            pooled = x.mean(dim=0)
        state_value = self.value_head(pooled).squeeze(-1)  # scalar

        return action_logits, state_value

    # ──────────────────────────────────────────────────────────────────────────

    def get_action(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedily select) an action and return associated quantities.

        Returns
        -------
        action   : int          — chosen node index
        log_prob : tensor scalar — log π(action | state)
        entropy  : tensor scalar — policy entropy (for exploration bonus)
        value    : tensor scalar — V(state)
        """
        logits, value = self.forward(node_features, adj, mask)
        probs = F.softmax(logits, dim=0)              # (N,)

        if deterministic:
            action = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = int(dist.sample().item())

        log_prob = torch.log(probs[action] + 1e-10)
        # Entropy computed only over valid actions
        valid_probs = probs[mask]
        entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum()

        return action, log_prob, entropy, value
