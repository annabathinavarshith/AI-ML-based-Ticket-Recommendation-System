AI/ML-based Ticket Recommendation System
📌 Overview

This project implements an AI/ML-powered ticket recommendation system that retrieves the most relevant historical support tickets for a given issue description.
It uses transformer-based semantic search with a vector database to enable fast and accurate similarity matching, helping reduce duplicate tickets and speed up issue resolution.

The system is designed as an internal intelligent search tool, inspired by Retrieval-Augmented Generation (RAG) architectures.

🚀 Key Features

Semantic similarity search using Sentence-BERT embeddings

High-performance vector search using FAISS

Cosine similarity–based ranking

Processes both ticket summary and description for better context understanding

Scalable to large ticket volumes

CLI-based interface for quick querying

Experimented with hybrid semantic + keyword search and finalized semantic-only approach for optimal accuracy

🧠 Architecture
User Query
   ↓
Sentence-BERT Encoder (all-mpnet-base-v2)
   ↓
FAISS Vector Index (Cosine Similarity)
   ↓
Top-K Similar Tickets

🛠️ Tech Stack

Programming: Python

ML / NLP: Sentence-BERT, Transformers (Hugging Face)

Vector Search: FAISS

Data Processing: Pandas, NumPy

Similarity Metrics: Cosine Similarity

Storage: CSV, NumPy, JSON

Tools: Git, GitLab, CLI