import numpy as np
from ml_lib.llm import CharTokenizer, Embedding

np.random.seed(42)

# 1. Sample training corpus
corpus = "hello world! building an llm from scratch."
tokenizer = CharTokenizer(corpus)
print(f"Vocab Size: {tokenizer.vocab_size} unique characters")

# 2. Encode a prompt into token IDs
prompt = "hello llm"
token_ids = tokenizer.encode(prompt)
print(f"Prompt: '{prompt}'")
print(f"Encoded IDs: {token_ids}")
print(f"Decoded back: '{tokenizer.decode(token_ids)}'")

# 3. Create Token & Positional Embedding tables
embed_dim = 8
max_seq_len = 32

token_embed_layer = Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=embed_dim)
pos_embed_layer = Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

# 4. Forward pass: combine Token Embeddings + Positional Embeddings
seq_len = len(token_ids)
positions = np.arange(seq_len)

tok_emb = token_embed_layer.forward(token_ids)   # Shape: (9, 8)
pos_emb = pos_embed_layer.forward(positions)   # Shape: (9, 8)
x = tok_emb + pos_emb                          # Shape: (9, 8)

print(f"\nToken Embedding shape:      {tok_emb.shape}")
print(f"Positional Embedding shape: {pos_emb.shape}")
print(f"Combined Input Matrix shape:{x.shape} (seq_len={seq_len}, embed_dim={embed_dim})")
