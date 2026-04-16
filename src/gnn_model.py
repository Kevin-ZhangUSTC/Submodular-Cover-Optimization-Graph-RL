"""
GNN-based policy and value networks for the sensor-selection RL problem.

Architecture
------------
Input  : node features  X  in R^{N x d}  and normalised adjacency  A  in R^{N x N}
Network:
    1. Linear input projection: X -> R^{N x H}
    2. K message-passing layers (GraphSAGE or GAT), each with optional residual.
    3. Policy head  : per-node linear -> scalar logit   (action selection)
    4. Value head   : attention-pooling or mean-pool over valid nodes -> scalar

GraphSAGE layer:
    h_new = LayerNorm( ReLU( Linear([h_self || h_agg]) ) + optional_residual )
    where h_agg = A * h (neighbourhood aggregation).
    When signed_adj=True it receives adj_pos and adj_neg and concatenates
    [h_self || A_pos*h || A_neg*h].

GAT (MultiHeadGraphAttentionLayer):
    For each head k:
        e_ij^k = LeakyReLU( a^k . [W^k*h_i || W^k*h_j] )
        alpha_ij^k = softmax_j( e_ij^k )  (masked by adj > 0)
        h_i^k = ELU( sum_j alpha_ij^k * W^k*h_j )
    Concat heads, project back to out_dim, LayerNorm, optional residual.

The policy outputs a masked softmax over unselected nodes; the value head
estimates V(s) used as the REINFORCE baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# GraphSAGE layer
# ==============================================================================

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE-style message-passing layer with optional residual.

    h_new = LayerNorm( ReLU( W * [h_self || A*h_neigh] ) + residual )

    When ``signed_adj=True`` the layer expects separate positive- and
    negative-part adjacency matrices and concatenates
    [h_self || A_pos*h || A_neg*h] (3*in_dim -> out_dim).

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    use_residual : add a residual (skip) connection
    signed_adj   : expect signed adjacency split as (adj_pos, adj_neg)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_residual: bool = False,
        signed_adj: bool = False,
    ) -> None:
        super().__init__()
        self.signed_adj = signed_adj
        cat_factor = 3 if signed_adj else 2
        self.linear = nn.Linear(in_dim * cat_factor, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.use_residual = use_residual
        if use_residual:
            self.res_proj = (
                nn.Identity() if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )
        else:
            self.res_proj = None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj_neg: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (N, in_dim)
        adj     : (N, N) row-normalised adjacency (unsigned or positive-part when signed)
        adj_neg : (N, N) negative-part adjacency (only used when signed_adj=True)

        Returns
        -------
        h : (N, out_dim)
        """
        h_neigh = adj @ x                                   # (N, in_dim)
        if self.signed_adj:
            if adj_neg is None:
                raise ValueError("adj_neg required when signed_adj=True")
            h_neg = adj_neg @ x                             # (N, in_dim)
            h_cat = torch.cat([x, h_neigh, h_neg], dim=-1) # (N, 3*in_dim)
        else:
            h_cat = torch.cat([x, h_neigh], dim=-1)        # (N, 2*in_dim)
        out = F.relu(self.linear(h_cat))
        if self.use_residual and self.res_proj is not None:
            out = out + self.res_proj(x)
        return self.norm(out)


# ==============================================================================
# Multi-head graph attention layer
# ==============================================================================

class MultiHeadGraphAttentionLayer(nn.Module):
    """Multi-head graph attention (GAT-style) layer with optional residual.

    For each head k:
        e_ij^k = LeakyReLU( a^k . [W^k*h_i || W^k*h_j] )
        alpha_ij^k = softmax_{j in N(i)}( e_ij^k )
        h_i^k = ELU( sum_j alpha_ij^k * W^k*h_j )
    Concat heads, project to out_dim, LayerNorm, optional residual.

    Attention is computed between every pair of nodes that shares a non-zero
    edge in the (unsigned) adjacency, plus self-loops.

    Parameters
    ----------
    in_dim      : input feature dimension
    out_dim     : output feature dimension
    n_heads     : number of attention heads
    dropout     : dropout rate on attention weights
    use_residual: add a residual (skip) connection
    signed_adj  : accept signed adjacency (adj_pos, adj_neg); edge mask built
                  from adj_pos + adj_neg > 0
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = False,
        signed_adj: bool = False,
    ) -> None:
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.signed_adj = signed_adj

        # Per-head weight matrices
        self.W = nn.ModuleList(
            [nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(n_heads)]
        )
        # Per-head attention vector (2*head_dim)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.empty(2 * self.head_dim)) for _ in range(n_heads)]
        )
        for param_a in self.a:
            nn.init.xavier_uniform_(param_a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.use_residual = use_residual
        if use_residual:
            self.res_proj = (
                nn.Identity() if in_dim == out_dim
                else nn.Linear(in_dim, out_dim, bias=False)
            )
        else:
            self.res_proj = None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj_neg: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (N, in_dim)
        adj     : (N, N) adjacency (edge mask / positive-part)
        adj_neg : (N, N) negative-part adjacency (only used when signed_adj=True)

        Returns
        -------
        h : (N, out_dim)
        """
        N = x.size(0)
        # Build edge mask: attend where abs(adj) > 0
        if self.signed_adj and adj_neg is not None:
            edge_mask = (adj + adj_neg) > 0              # (N, N)
        else:
            edge_mask = adj > 0                          # (N, N)
        # Include self-loops so each node attends to itself
        self_eye = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_mask = edge_mask | self_eye

        head_outputs = []
        for k in range(self.n_heads):
            Wx = self.W[k](x)                            # (N, head_dim)
            a_src = (Wx * self.a[k][:self.head_dim]).sum(dim=-1, keepdim=True)   # (N,1)
            a_dst = (Wx * self.a[k][self.head_dim:]).sum(dim=-1, keepdim=True)   # (N,1)
            e = self.leaky_relu(a_src + a_dst.T)         # (N, N) broadcast
            e = e.masked_fill(~edge_mask, float("-inf"))
            alpha = torch.softmax(e, dim=1)              # (N, N)
            alpha = self.dropout(alpha)
            h_k = F.elu(alpha @ Wx)                      # (N, head_dim)
            head_outputs.append(h_k)

        out = torch.cat(head_outputs, dim=-1)            # (N, out_dim)
        if self.use_residual and self.res_proj is not None:
            out = out + self.res_proj(x)
        return self.norm(out)


# ==============================================================================
# Attention-pooling value head helper
# ==============================================================================

class AttentionPooling(nn.Module):
    """Attention-weighted pooling over node embeddings.

    beta_i = softmax( v^T * tanh( W_pool * h_i ) )
    z = sum_i beta_i * h_i

    Parameters
    ----------
    hidden_dim : dimension of node embeddings
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_pool = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.v.unsqueeze(0))

    def forward(
        self, x: torch.Tensor, mask: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (N, hidden_dim)
        mask : (N,) bool -- True for nodes to include (e.g., unselected)

        Returns
        -------
        z : (hidden_dim,) pooled representation
        """
        if mask is not None and mask.any():
            x_valid = x[mask]
        else:
            x_valid = x
        scores = torch.tanh(self.W_pool(x_valid)) @ self.v   # (N_valid,)
        beta = torch.softmax(scores, dim=0)                   # (N_valid,)
        return (beta.unsqueeze(-1) * x_valid).sum(dim=0)      # (hidden_dim,)


# ==============================================================================
# Main GNN policy
# ==============================================================================

class GNNPolicy(nn.Module):
    """Actor-Critic GNN network for sequential sensor selection.

    Parameters
    ----------
    node_feat_dim : int
        Number of input features per node (default 5).
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of GNN message-passing layers.
    layer_type : {"sage", "gat"}
        Which GNN layer to use.
    n_heads : int
        Number of attention heads (GAT only).
    attention_dropout : float
        Dropout on GAT attention weights.
    use_residual : bool
        Add residual connections to each GNN layer.
    use_attention_pooling : bool
        Use attention-pooling instead of mean-pooling in the value head.
    signed_adj : bool
        Expect separate positive/negative adjacency matrices.
    """

    def __init__(
        self,
        node_feat_dim: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 3,
        layer_type: str = "sage",
        n_heads: int = 4,
        attention_dropout: float = 0.0,
        use_residual: bool = False,
        use_attention_pooling: bool = False,
        signed_adj: bool = False,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.signed_adj = signed_adj
        self.use_attention_pooling = use_attention_pooling

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        def _make_layer(in_d: int, out_d: int) -> nn.Module:
            if layer_type == "sage":
                return GraphSAGELayer(
                    in_d, out_d,
                    use_residual=use_residual,
                    signed_adj=signed_adj,
                )
            elif layer_type == "gat":
                return MultiHeadGraphAttentionLayer(
                    in_d, out_d,
                    n_heads=n_heads,
                    dropout=attention_dropout,
                    use_residual=use_residual,
                    signed_adj=signed_adj,
                )
            else:
                raise ValueError(
                    f"Unknown layer_type '{layer_type}'. Choose 'sage' or 'gat'."
                )

        self.gnn_layers = nn.ModuleList(
            [_make_layer(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Policy head: maps node embedding -> scalar action logit
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value head
        if use_attention_pooling:
            self.attn_pool: AttentionPooling | None = AttentionPooling(hidden_dim)
        else:
            self.attn_pool = None
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # --------------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: "torch.Tensor | None" = None,
        adj_neg: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        node_features : (N, feat_dim)
        adj           : (N, N) normalised adjacency (positive-part or unsigned)
        mask          : (N,) bool tensor -- True where action is valid (not yet selected)
        adj_neg       : (N, N) negative-part adjacency (signed_adj only)

        Returns
        -------
        action_logits : (N,)  -- raw logits (-inf for already-selected nodes)
        state_value   : scalar tensor
        """
        x = self.input_proj(node_features)            # (N, H)

        for layer in self.gnn_layers:
            x = layer(x, adj, adj_neg)                # (N, H)

        # -- action logits -------------------------------------------------------
        action_logits = self.policy_head(x).squeeze(-1)   # (N,)
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, float("-inf"))

        # -- state value (pool over valid / unselected nodes) --------------------
        if self.attn_pool is not None:
            pooled = self.attn_pool(x, mask)          # (H,)
        elif mask is not None and mask.any():
            pooled = x[mask].mean(dim=0)              # (H,)
        else:
            pooled = x.mean(dim=0)
        state_value = self.value_head(pooled).squeeze(-1)  # scalar

        return action_logits, state_value

    # --------------------------------------------------------------------------

    def get_action(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        adj_neg: "torch.Tensor | None" = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or greedily select) an action and return associated quantities.

        Returns
        -------
        action   : int          -- chosen node index
        log_prob : tensor scalar -- log pi(action | state)
        entropy  : tensor scalar -- policy entropy (for exploration bonus)
        value    : tensor scalar -- V(state)
        """
        logits, value = self.forward(node_features, adj, mask, adj_neg)
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
