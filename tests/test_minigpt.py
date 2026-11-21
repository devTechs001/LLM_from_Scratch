import numpy as np
import pytest
from ml_lib.llm import CharTokenizer, MiniGPT
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient


def test_minigpt_convergence_and_generation():
    """Verify MiniGPT trains on corpus and generates coherent continuation."""
    np.random.seed(42)
    text = "to be or not to be, that is the question. building an llm from scratch works!"
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    X_train = data[:-1]
    y_train = data[1:]

    model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=32, max_seq_len=len(text))

    learning_rate = 0.05
    epochs = 400

    for epoch in range(epochs):
        logits = model.forward(X_train)
        probs = softmax(logits)
        loss = cross_entropy_loss(y_train, probs)

        d_logits = cross_entropy_softmax_gradient(y_train, probs)
        model.backward(d_logits, learning_rate)

    final_loss = cross_entropy_loss(y_train, softmax(model.forward(X_train)))
    assert final_loss < 0.25, f"MiniGPT final loss {final_loss:.4f} too high!"

    # Test generation
    prompt = "to be"
    prompt_ids = tokenizer.encode(prompt)
    output_ids = model.generate(prompt_ids, max_new_tokens=20)
    generated_text = tokenizer.decode(output_ids)
    assert generated_text.startswith("to be or not to be")


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Training text corpus
    text = "to be or not to be, that is the question. building an llm from scratch works!"
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    # Input is tokens [0 .. N-2], Target is next tokens [1 .. N-1]
    X_train = data[:-1]
    y_train = data[1:]

    # 2. Initialize MiniGPT
    model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=32, max_seq_len=len(text))

    # 3. Training loop
    learning_rate = 0.05
    epochs = 600

    for epoch in range(epochs):
        logits = model.forward(X_train)
        probs = softmax(logits)
        loss = cross_entropy_loss(y_train, probs)
        
        d_logits = cross_entropy_softmax_gradient(y_train, probs)
        model.backward(d_logits, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    # 4. Generate text from a short prompt!
    prompt = "to be"
    prompt_ids = tokenizer.encode(prompt)
    output_ids = model.generate(prompt_ids, max_new_tokens=60)
    print("\nPrompt:   ", prompt)
    print("Generated:", tokenizer.decode(output_ids))
