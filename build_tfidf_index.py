import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
CSV_PATH = "data/tickets--defects.csv"
TFIDF_DIR = "tfidf_store"
os.makedirs(TFIDF_DIR, exist_ok=True)

# Load tickets
df = pd.read_csv(CSV_PATH, low_memory=False)
df.fillna("", inplace=True)
df["combined"] = df[["Summary", "Description"]].agg(" ".join, axis=1)

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=50000)
tfidf_matrix = vectorizer.fit_transform(df["combined"].tolist())

# Save vectorizer and matrix
with open(os.path.join(TFIDF_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(TFIDF_DIR, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("[✅] TF-IDF index built and saved.")
