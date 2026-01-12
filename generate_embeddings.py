import pandas as pd
import json
import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch

# Paths
CSV_PATH = "data/tickets--defects.csv"
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 🧹 Optional text cleaner
def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()

# 1. Load ticket CSV
df = pd.read_csv(CSV_PATH, low_memory=False)

# 2. Extract text fields (customize this based on your data)
text_columns = ["Summary", "Description"]
df.fillna("", inplace=True)

# 3. Concatenate selected fields
df["combined"] = df[text_columns].agg(" ".join, axis=1)
df = df.copy()

# 🎯 Initialize model with GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Load embedding model
model = SentenceTransformer("all-mpnet-base-v2", device=device)

# 5. Generate embeddings
sentences = df["combined"].tolist()
embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

# 6. Save embeddings
np.save(os.path.join(EMBEDDING_DIR, "ticket_embeddings.npy"), embeddings)


# 7. Save mapping of ID ↔ index
id_map = {str(i): df.iloc[i]["Ticket key"] for i in range(len(df))}
with open(os.path.join(EMBEDDING_DIR, "id_map.json"), "w") as f:
   # Ensure JSON serializable types
   id_map_clean = {str(k): str(v) for k, v in id_map.items()}
   json.dump(id_map_clean, f, indent=2)


print("✅ Embeddings and ID map saved.")






