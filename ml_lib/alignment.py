"""Alignment and Preference Optimization from scratch in pure NumPy.

Includes:
- RewardModel: Scalar reward head for Bradley-Terry preference modeling.
- pairwise_reward_loss: Pairwise ranking loss -log(sigma(r_w - r_l)) with exact gradients.
- Direct Preference Optimization (DPO): Closed-form RLHF objective bypassing
  an explicit reward model via implicit reward margins:
  L_DPO = -E [ log sigma( beta * log(pi(y_w|x)/ref(y_w|x)) - beta * log(pi(y_l|x)/ref(y_l|x)) ) ]
"""

from typing import Tuple, Optional
import numpy as np
from ml_lib.nn import Module, Linear
from ml_lib.losses import sigmoid, softmax


class RewardModel(Module):
    """Transformer-based Reward Model with a scalar linear projection head.

    Given a sequence of token representations (B, T, D), extracts the representation
    at the last token position (EOS token) and projects to a scalar reward r in R.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.head = Linear(in_features, 1)
        self._cached_last_tokens = None
        self._cached_seq_len = None

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            hidden_states: (B, T, D) or (B, D)

        Returns:
            rewards: (B, 1) scalar rewards per sequence.
        """
        if hidden_states.ndim == 3:
            B, T, D = hidden_states.shape
            self._cached_seq_len = T
            # Last token representation
            last_tokens = hidden_states[:, -1, :]  # (B, D)
        else:
            self._cached_seq_len = 1
            last_tokens = hidden_states

        self._cached_last_tokens = last_tokens
        return self.head.forward(last_tokens)

    def backward(self, d_rewards: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """Backward pass w.r.t rewards.

        Args:
            d_rewards: (B, 1)

        Returns:
            d_hidden: (B, T, D) or (B, D)
        """
        d_last = self.head.backward(d_rewards, learning_rate=learning_rate)
        if self._cached_seq_len > 1:
            B, D = d_last.shape
            T = self._cached_seq_len
            d_hidden = np.zeros((B, T, D))
            d_hidden[:, -1, :] = d_last
            return d_hidden
        return d_last


def pairwise_reward_loss(
    r_chosen: np.ndarray,
    r_rejected: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Bradley-Terry pairwise ranking loss for reward model training.

    L = -mean( log sigma( r_chosen - r_rejected ) )

    Gradients:
        dL / d(r_chosen)   = sigma(r_rejected - r_chosen) - 1 = -(1 - sigma(r_chosen - r_rejected))
        dL / d(r_rejected) = 1 - sigma(r_rejected - r_chosen) = (1 - sigma(r_chosen - r_rejected))
    """
    diff = r_chosen - r_rejected  # (B, 1)
    probs = sigmoid(diff)
    probs_clamped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    loss = -float(np.mean(np.log(probs_clamped)))

    # Gradient: sigma(diff) - 1 for chosen, 1 - sigma(diff) for rejected
    B = r_chosen.shape[0]
    grad_chosen = (probs - 1.0) / B
    grad_rejected = (1.0 - probs) / B

    return loss, grad_chosen, grad_rejected


def compute_sequence_log_probs(
    logits: np.ndarray,
    target_ids: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sum of log probabilities of target tokens given logits.

    Args:
        logits: (B, T, vocab_size)
        target_ids: (B, T)
        mask: (B, T) binary mask where 1 = active response token, 0 = prompt/padding.

    Returns:
        sum_log_probs: (B,) sum of log probabilities for each sample.
        token_probs: (B, T, vocab_size) softmax probabilities.
    """
    B, T, V = logits.shape
    token_probs = np.zeros_like(logits)
    sum_log_probs = np.zeros(B)

    if mask is None:
        mask = np.ones((B, T), dtype=bool)

    for b in range(B):
        sample_log_prob = 0.0
        for t in range(T):
            p = softmax(logits[b, t])
            token_probs[b, t] = p
            if mask[b, t]:
                token_id = target_ids[b, t]
                sample_log_prob += np.log(np.clip(p[token_id], 1e-15, 1.0))
        sum_log_probs[b] = sample_log_prob

    return sum_log_probs, token_probs


def dpo_loss_and_gradients(
    pi_chosen_logits: np.ndarray,
    pi_rejected_logits: np.ndarray,
    ref_chosen_logits: np.ndarray,
    ref_rejected_logits: np.ndarray,
    chosen_ids: np.ndarray,
    rejected_ids: np.ndarray,
    chosen_mask: Optional[np.ndarray] = None,
    rejected_mask: Optional[np.ndarray] = None,
    beta: float = 0.1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Direct Preference Optimization (DPO) objective & analytic policy gradients.

    L_DPO = -E [ log sigma( beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)) ) ]

    Args:
        pi_chosen_logits: (B, T_w, V) policy model logits on chosen sequence
        pi_rejected_logits: (B, T_l, V) policy model logits on rejected sequence
        ref_chosen_logits: (B, T_w, V) frozen reference model logits on chosen
        ref_rejected_logits: (B, T_l, V) frozen reference model logits on rejected
        chosen_ids: (B, T_w) target token IDs
        rejected_ids: (B, T_l) target token IDs
        chosen_mask: (B, T_w) response token mask
        rejected_mask: (B, T_l) response token mask
        beta: DPO temperature hyperparameter (typical 0.1 - 0.5)

    Returns:
        loss: Scalar mean DPO loss.
        grad_pi_chosen: (B, T_w, V) gradient w.r.t policy chosen logits.
        grad_pi_rejected: (B, T_l, V) gradient w.r.t policy rejected logits.
    """
    B = pi_chosen_logits.shape[0]

    # Compute sequence log probabilities
    log_pi_w, probs_pi_w = compute_sequence_log_probs(pi_chosen_logits, chosen_ids, chosen_mask)
    log_pi_l, probs_pi_l = compute_sequence_log_probs(pi_rejected_logits, rejected_ids, rejected_mask)
    log_ref_w, _ = compute_sequence_log_probs(ref_chosen_logits, chosen_ids, chosen_mask)
    log_ref_l, _ = compute_sequence_log_probs(ref_rejected_logits, rejected_ids, rejected_mask)

    # Implicit rewards
    # r_w = beta * (log_pi_w - log_ref_w)
    # r_l = beta * (log_pi_l - log_ref_l)
    r_w = beta * (log_pi_w - log_ref_w)
    r_l = beta * (log_pi_l - log_ref_l)
    margin = r_w - r_l  # (B,)

    # DPO Loss = -log sigma(margin)
    sig_margin = sigmoid(margin)
    loss = -float(np.mean(np.log(np.clip(sig_margin, 1e-15, 1.0))))

    # Scalar gradient factor: -beta * sigma(-margin) / B
    # Since sigma(-margin) = 1 - sigma(margin)
    weight = (1.0 - sig_margin) * (beta / B)  # (B,)

    # Gradients w.r.t policy logits
    # For chosen: dL / dz_{t, k} = -weight * (1[k == target] - p_k) = weight * (p_k - 1[k == target])
    # For rejected: dL / dz_{t, k} = +weight * (1[k == target] - p_k) = -weight * (p_k - 1[k == target])
    grad_pi_chosen = np.zeros_like(pi_chosen_logits)
    grad_pi_rejected = np.zeros_like(pi_rejected_logits)

    if chosen_mask is None:
        chosen_mask = np.ones((B, pi_chosen_logits.shape[1]), dtype=bool)
    if rejected_mask is None:
        rejected_mask = np.ones((B, pi_rejected_logits.shape[1]), dtype=bool)

    for b in range(B):
        w_b = weight[b]
        # Chosen gradients
        for t in range(pi_chosen_logits.shape[1]):
            if chosen_mask[b, t]:
                target = chosen_ids[b, t]
                grad_pi_chosen[b, t] = probs_pi_w[b, t] * w_b
                grad_pi_chosen[b, t, target] -= w_b

        # Rejected gradients
        for t in range(pi_rejected_logits.shape[1]):
            if rejected_mask[b, t]:
                target = rejected_ids[b, t]
                grad_pi_rejected[b, t] = -probs_pi_l[b, t] * w_b
                grad_pi_rejected[b, t, target] += w_b

    return loss, grad_pi_chosen, grad_pi_rejected
