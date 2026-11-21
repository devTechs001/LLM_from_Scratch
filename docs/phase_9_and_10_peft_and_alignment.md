# Phase 9 & 10: Parameter-Efficient Fine-Tuning (PEFT / LoRA) & Alignment (DPO & Reward Modeling)

## Executive Summary

Phases 9 and 10 introduce post-pretraining adaptation and human alignment primitives built entirely in pure NumPy without third-party frameworks:
1. **Phase 9: Parameter-Efficient Fine-Tuning (PEFT / LoRA & SFT)**:
   - Low-Rank Adaptation (`LoRALinear`): Decomposes weight updates $\Delta W = \frac{\alpha}{r} B A$, freezing original parameters $W_0$ and training $< 10\%$ of parameters.
   - Zero-overhead inference weight merging ($W \leftarrow W_0 + \frac{\alpha}{r} (B A)^T$) and dynamic unmerging.
   - Supervised Fine-Tuning (`sft_masked_cross_entropy`): Prompt-completion target masking computing loss and gradients strictly on response tokens.
2. **Phase 10: Alignment & Preference Optimization (Reward Modeling & DPO)**:
   - Bradley-Terry Reward Model (`RewardModel` & `pairwise_reward_loss`): Scalar reward scoring head trained with pairwise preference ranking loss $-\log \sigma(r_w - r_l)$.
   - Direct Preference Optimization (`dpo_loss_and_gradients`): Closed-form RLHF objective bypassing reward modeling via implicit policy margins $\beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$ with exact analytic policy gradients.

---

## 1. Phase 9: Low-Rank Adaptation (LoRA)

### 1.1 Mathematical Formulation
Standard full parameter fine-tuning updates all weights $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$. LoRA hypothesizes that weight updates have low "intrinsic rank" $r \ll \min(d_{\text{in}}, d_{\text{out}})$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} A^T B^T$$

where:
- $W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is frozen.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized from $\mathcal{N}\left(0, \frac{1}{r}\right)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to $0$, ensuring $\Delta W = 0$ at initialization.
- $\frac{\alpha}{r}$ is a constant scaling hyperparameter.

### 1.2 Forward Pass
For input activation $X \in \mathbb{R}^{B \times T \times d_{\text{in}}}$:

$$Y = X W_0 + \frac{\alpha}{r} (X A^T) B^T + b$$

### 1.3 Backward Gradients
Given incoming gradient $\nabla_Y \mathcal{L} \in \mathbb{R}^{B \times T \times d_{\text{out}}}$:

$$\nabla_B \mathcal{L} = \frac{\alpha}{r} (\nabla_Y \mathcal{L})^T (X A^T)$$
$$\nabla_{X A^T} \mathcal{L} = \frac{\alpha}{r} (\nabla_Y \mathcal{L}) B$$
$$\nabla_A \mathcal{L} = (\nabla_{X A^T} \mathcal{L})^T X$$
$$\nabla_X \mathcal{L}_{\text{LoRA}} = (\nabla_{X A^T} \mathcal{L}) A$$

### 1.4 Weight Merging for Zero-Overhead Inference
When serving in production:
$$W_0 \leftarrow W_0 + \frac{\alpha}{r} (B A)^T$$
The LoRA branch is folded directly into the base weights, yielding zero additional latency or FLOPs during generation.

---

## 2. Supervised Fine-Tuning (SFT) with Prompt Masking

During instruction tuning, the model should only be penalized for predicting output completions, not user prompts.

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{N_{\text{resp}}} \sum_{t \ge L_{\text{prompt}}} \log P(w_{t} \mid w_{<t})$$

Positions $t < L_{\text{prompt}} - 1$ receive an explicit gradient of zero:
$$\nabla_{z_t} \mathcal{L}_{\text{SFT}} = 0 \quad \forall t < L_{\text{prompt}} - 1$$

---

## 3. Phase 10: Alignment & Direct Preference Optimization (DPO)

### 3.1 Pairwise Bradley-Terry Reward Modeling
Given a preferred completion $y_w$ and rejected completion $y_l$ for prompt $x$:
$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$
$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]$$

### 3.2 Direct Preference Optimization (DPO)
DPO mathematically reparameterizes the RLHF objective (under KL regularization w.r.t reference model $\pi_{\text{ref}}$) to directly express the reward in terms of policy probabilities:
$$r^*(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

The DPO loss is:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Let the implicit margin be:
$$\hat{h}_\theta = \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

### 3.3 Analytic Policy Logit Gradients
The gradient w.r.t the logits $z_{t, k}$ of the policy model is:
$$\frac{\partial \mathcal{L}_{\text{DPO}}}{\partial z_{w, t, k}} = \beta \sigma(-\hat{h}_\theta) (p_{w, t, k} - \mathbb{I}(k = y_{w, t}))$$
$$\frac{\partial \mathcal{L}_{\text{DPO}}}{\partial z_{l, t, k}} = -\beta \sigma(-\hat{h}_\theta) (p_{l, t, k} - \mathbb{I}(k = y_{l, t}))$$

- When $\hat{h}_\theta \gg 0$ (the model already strongly prefers the winning response), $\sigma(-\hat{h}_\theta) \to 0$ and gradients vanish.
- When the model mistakenly favors the rejected response ($\hat{h}_\theta < 0$), $\sigma(-\hat{h}_\theta) \to 1$, applying maximum corrective gradient pressure to increase winning token likelihood and suppress losing tokens.

---

## 4. Empirical Test Suite & Results

Across `tests/test_phase9_and_10_alignment.py`:
- `test_lora_initialization_zero_delta`: Initial identity equivalence (**PASSED**)
- `test_lora_forward_backward_shapes`: 3D tensor backpropagation & low-rank gradients (**PASSED**)
- `test_lora_weight_merging_and_unmerging`: Exact numerical equivalence before and after folding (**PASSED**)
- `test_sft_masked_cross_entropy`: Strict prompt gradient masking (**PASSED**)
- `test_reward_model_pairwise_loss`: Bradley-Terry reward scoring and gradient symmetry (**PASSED**)
- `test_dpo_loss_and_gradients`: Margin optimization and correct policy gradient directions (**PASSED**)
- `test_lora_finetuning_convergence`: Task adaptation reducing loss $> 60\%$ while freezing base weights (**PASSED**)

**64 out of 64 unit tests pass repository-wide** in 2.11s.
