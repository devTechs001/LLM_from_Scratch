"""Sparse Mixture-of-Experts (MoE) with Top-k Gating & Auxiliary Load Balancing from scratch in pure NumPy.

Includes:
- SparseMoE: Token-level sparse mixture of feed-forward experts.
- Top-k Softmax Gating router with load-balancing auxiliary loss (Switch Transformer / GShard).
- Forward routing and backpropagation through activated experts and router.
"""

from typing import Tuple, List, Optional
import numpy as np
from ml_lib.nn import Module, Linear, GELU
from ml_lib.losses import softmax


class ExpertMLP(Module):
    """Feed-Forward Expert network: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.act = GELU()
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(x)
        h = self.act.forward(h)
        return self.fc2.forward(h)

    def backward(self, d_out: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        d_h = self.fc2.backward(d_out, learning_rate=learning_rate)
        d_act = self.act.backward(d_h)
        return self.fc1.backward(d_act, learning_rate=learning_rate)


class SparseMoE(Module):
    """Sparse Mixture-of-Experts (MoE) layer with Top-k Gating router."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coeff = aux_loss_coeff

        # Router linear gate: d_model -> num_experts
        self.gate = Linear(d_model, num_experts)

        # Expert pool
        self.experts = [ExpertMLP(d_model, d_ff) for _ in range(num_experts)]

        # Cache for backward pass
        self._cached_x = None
        self._cached_router_probs = None
        self._cached_sparse_weights = None
        self._cached_expert_outputs = {}
        self.aux_loss = 0.0

    def compute_auxiliary_loss(
        self,
        router_probs: np.ndarray,
        topk_indices: np.ndarray
    ) -> float:
        """Compute GShard / Switch Transformer load balancing auxiliary loss:

        L_aux = aux_loss_coeff * num_experts * sum_{i=1}^{num_experts} (f_i * P_i)
        where:
            f_i = fraction of tokens routed to expert i
            P_i = mean probability allocated to expert i across all tokens
        """
        N_tokens = router_probs.shape[0]
        # P_i: mean router probability for expert i
        P = np.mean(router_probs, axis=0)  # (num_experts,)

        # f_i: fraction of tokens where expert i is in top-k
        expert_counts = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            expert_counts[i] = np.sum(np.any(topk_indices == i, axis=1))
        f = expert_counts / max(N_tokens, 1)

        aux_loss = self.aux_loss_coeff * self.num_experts * float(np.sum(f * P))
        return aux_loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass routing tokens to top-k experts.

        Args:
            x: (B, T, d_model) or (N, d_model)

        Returns:
            out: Same shape as x with blended expert outputs.
        """
        self._cached_x = x
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # (N, d_model)
        N = x_flat.shape[0]

        # 1. Router logits & probabilities
        router_logits = self.gate.forward(x_flat)  # (N, num_experts)
        router_probs = softmax(router_logits)      # (N, num_experts)
        self._cached_router_probs = router_probs

        # 2. Top-k gating selection
        topk_indices = np.argsort(-router_logits, axis=-1)[:, :self.top_k]  # (N, top_k)

        # Build sparse routing matrix R: (N, num_experts)
        sparse_weights = np.zeros((N, self.num_experts))
        for row in range(N):
            selected = topk_indices[row]
            selected_probs = router_probs[row, selected]
            sum_prob = np.sum(selected_probs)
            norm_probs = selected_probs / (sum_prob + 1e-12)
            for k_idx, exp_idx in enumerate(selected):
                sparse_weights[row, exp_idx] = norm_probs[k_idx]
        self._cached_sparse_weights = sparse_weights

        # 3. Auxiliary load balancing loss
        self.aux_loss = self.compute_auxiliary_loss(router_probs, topk_indices)

        # 4. Dispatch tokens to active experts (called at most once per expert)
        out_flat = np.zeros_like(x_flat)
        self._cached_expert_outputs = {}

        for exp_idx in range(self.num_experts):
            mask = (sparse_weights[:, exp_idx] > 0)
            if np.any(mask):
                tokens = x_flat[mask]
                exp_out = self.experts[exp_idx].forward(tokens)
                self._cached_expert_outputs[exp_idx] = exp_out
                weights = sparse_weights[mask, exp_idx, np.newaxis]
                out_flat[mask] += weights * exp_out

        return out_flat.reshape(orig_shape)

    def backward(self, d_out: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """Backward pass propagating gradients through active experts and router."""
        x = self._cached_x
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        d_out_flat = d_out.reshape(-1, self.d_model)
        N = x_flat.shape[0]

        sparse_weights = self._cached_sparse_weights
        router_probs = self._cached_router_probs

        d_x_flat = np.zeros_like(x_flat)
        d_router_logits = np.zeros((N, self.num_experts))

        # 1. Backprop through experts
        for exp_idx in range(self.num_experts):
            mask = (sparse_weights[:, exp_idx] > 0)
            if np.any(mask):
                d_tokens = d_out_flat[mask]
                weights = sparse_weights[mask, exp_idx, np.newaxis]
                # Backward through expert once
                d_exp_in = self.experts[exp_idx].backward(
                    d_tokens * weights,
                    learning_rate=learning_rate
                )
                d_x_flat[mask] += d_exp_in

                # Gradient w.r.t routing weight: d_out * exp_out
                if exp_idx in self._cached_expert_outputs:
                    exp_out = self._cached_expert_outputs[exp_idx]
                    d_weight = np.sum(d_tokens * exp_out, axis=1)  # (N_active,)
                    d_router_logits[mask, exp_idx] += d_weight

        # 2. Backprop router gate
        # Router logits softmax gradient
        d_router = (d_router_logits - np.sum(d_router_logits * router_probs, axis=1, keepdims=True)) * router_probs
        d_x_router = self.gate.backward(d_router, learning_rate=learning_rate)
        d_x_flat += d_x_router

        return d_x_flat.reshape(orig_shape)
