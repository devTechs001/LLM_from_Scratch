import numpy as np
from ml_lib.llm import CharTokenizer, MiniGPT
from ml_lib.losses import softmax, cross_entropy_loss, cross_entropy_softmax_gradient

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
