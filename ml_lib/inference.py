import numpy as np
from typing import Optional, List, Generator, Any
from ml_lib.losses import softmax

# =====================================================================
# Token Sampling Logic
# =====================================================================

def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    past_token_ids: Optional[List[int]] = None,
) -> int:
    """
    Samples the next token ID from raw logits using:
      1. Repetition penalty (penalizes recently generated tokens).
      2. Temperature scaling (controls randomness/entropy).
      3. Top-k filtering (truncates to top-k highest probability tokens).
      4. Top-p (nucleus) filtering (truncates to cumulative probability mass >= p).
    """
    logits = logits.copy().astype(float)

    # 1. Repetition penalty
    if past_token_ids and repetition_penalty != 1.0:
        for tok in set(past_token_ids):
            if tok < len(logits):
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty

    # 2. Greedy decoding on zero/near-zero temperature
    if temperature <= 1e-5:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # 3. Top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, len(logits))
        cutoff = np.sort(logits)[-top_k]
        logits[logits < cutoff] = -1e9

    # Convert to probabilities with stable softmax
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)

    # 4. Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Mask out tokens beyond cumulative threshold p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least the single highest probability token
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0

        # Renormalize
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs = probs / sum_p
        else:
            probs[sorted_indices[0]] = 1.0

    # 5. Stochastic multinomial sampling
    return int(np.random.choice(len(probs), p=probs))


# =====================================================================
# Generation Pipeline / Inference Engine
# =====================================================================

class GenerationPipeline:
    """
    Production-grade inference engine supporting:
      - Autoregressive generation with KV caching
      - Full sampling controls (temperature, top_k, top_p, repetition_penalty)
      - Streaming generation generator
      - Stop tokens / early termination
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_words: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> str:
        """Generates full completed text starting from a prompt string."""
        generated_chunks = list(
            self.stream_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_words=stop_words,
                use_cache=use_cache,
            )
        )
        return prompt + "".join(generated_chunks)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_words: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Generator[str, None, None]:
        """
        Streaming token generator: yields newly predicted characters/tokens in real-time.
        """
        prompt_ids = list(self.tokenizer.encode(prompt))
        generated_ids = list(prompt_ids)

        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

        # Prefill phase
        if use_cache and hasattr(self.model, "forward"):
            prompt_arr = np.array(prompt_ids, dtype=int)
            logits = self.model.forward(prompt_arr, use_cache=True, start_pos=0)
            last_logits = logits[-1]
            next_id = sample_next_token(
                last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_token_ids=generated_ids,
            )
            generated_ids.append(next_id)
            char = self.tokenizer.decode([next_id])
            yield char

            # Decode phase
            for _ in range(1, max_new_tokens):
                curr_tok = np.array([generated_ids[-1]], dtype=int)
                curr_pos = len(generated_ids) - 1
                logits = self.model.forward(curr_tok, use_cache=True, start_pos=curr_pos)
                last_logits = logits[0] if logits.ndim == 2 else logits
                next_id = sample_next_token(
                    last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_token_ids=generated_ids,
                )
                generated_ids.append(next_id)
                char = self.tokenizer.decode([next_id])
                yield char

                # Early stopping check
                if stop_words:
                    accumulated_text = self.tokenizer.decode(generated_ids[len(prompt_ids):])
                    if any(stop_w in accumulated_text for stop_w in stop_words):
                        break
        else:
            # Non-cached generation fallback
            for _ in range(max_new_tokens):
                context = np.array(generated_ids[-self.model.max_seq_len:], dtype=int)
                logits = self.model.forward(context)
                last_logits = logits[-1]
                next_id = sample_next_token(
                    last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_token_ids=generated_ids,
                )
                generated_ids.append(next_id)
                char = self.tokenizer.decode([next_id])
                yield char

                if stop_words:
                    accumulated_text = self.tokenizer.decode(generated_ids[len(prompt_ids):])
                    if any(stop_w in accumulated_text for stop_w in stop_words):
                        break
