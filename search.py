import faiss
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle

# --------------------
# Config
# --------------------
TOP_K = 10
SEMANTIC_THRESHOLD = 0.40  # Fallback if highest similarity is below thi

# --------------------
# Paths
# --------------------
TFIDF_VECTORIZER_PATH = "tfidf_store/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = "tfidf_store/tfidf_matrix.pkl"

# Load FAISS index
faiss_index = faiss.read_index("faiss_store/index.faiss")

# Load ID map
with open("embeddings/id_map.json", "r") as f:
    id_map = json.load(f)
df = pd.read_csv("data/tickets.csv", low_memory=False)

# Load Sentence-BERT model
model = SentenceTransformer("all-mpnet-base-v2")

# Load TF-IDF vectorizer + matrix
with open(TFIDF_VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open(TFIDF_MATRIX_PATH, "rb") as f:
    tfidf_matrix = pickle.load(f)

def semantic_search(query, top_k=TOP_K):
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        ticket_key = id_map[str(idx)]
        summary_row = df[df["Ticket key"] == ticket_key]
        summary = summary_row["Summary"].values[0] if not summary_row.empty else "Summary not found"
        results.append((ticket_key, summary, dist))
    
    return results

# --------------------
# Keyword (TF-IDF) Search
# --------------------
def keyword_search(query, top_k=TOP_K):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        ticket_key = df.iloc[idx]["Ticket key"]
        summary = df.iloc[idx]["Summary"]
        score = cosine_sim[idx]
        results.append((ticket_key, summary, score))
    return results


# --------------------
# Hybrid Search
# --------------------
def search_similar_tickets(query):
    sem_results = semantic_search(query)
    
    if not sem_results or sem_results[0][2] < SEMANTIC_THRESHOLD:
        print("🔁 Semantic match is weak — switching to keyword-based search...")
        return keyword_search(query, TOP_K)
    
    return sem_results;	


# CLI usage
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Enter your issue/query (or 'exit'): ")
        if query.lower() == "exit":
            break
        results = search_similar_tickets(query)
        print("\nTop Matches:")
        for ticket, summary, score in results:
            print(f"🎫 Ticket: {ticket} | 📝 Summary: {summary} | 🔗 Similarity Score: {score:.4f}")
