# scripts/build_faiss_index.py

import os
import faiss
import numpy as np
import json
from sklearn.preprocessing import normalize

# Paths
EMBEDDINGS_PATH = "embeddings/ticket_embeddings.npy"
ID_MAP_PATH = "embeddings/id_map.json"
FAISS_INDEX_DIR = "faiss_store"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")

# Step 1: Load embeddings and ID map
embeddings = np.load(EMBEDDINGS_PATH)
with open(ID_MAP_PATH, 'r') as f:
    id_map = json.load(f)

# Normalize to unit vectors (important for cosine similarity)
embeddings = normalize(embeddings, axis=1)

# Step 2: Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # IP = inner product ≈ cosine similarity (after normalization)
index.add(embeddings)

# Step 3: Save the index
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)

print(f"[✅] FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"[ℹ️] Total vectors indexed: {index.ntotal}")