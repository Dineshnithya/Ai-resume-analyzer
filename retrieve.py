# retrieve.py

from sentence_transformers import SentenceTransformer
import numpy as np
from vector_store import VectorStore

def retrieve(query: str, top_k: int = 3):
    """
    Embed the query and retrieve the top_k relevant documents from ChromaDB.
    Returns a list of (document, distance) tuples.
    """
    # Load embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Embed the query
    q_embed = model.encode([query])
    
    # Query vector store
    store = VectorStore()
    results = store.query(query_embeds=[q_embed.tolist()], n_results=top_k)
    
    docs = results["documents"][0]
    dists = results["distances"][0]
    
    # Pair up doc with its distance
    return list(zip(docs, dists))

# Example usage:
# hits = retrieve("What is FastAPI?", top_k=2)
# for doc, dist in hits:
#     print(f"Distance={dist:.3f}: {doc[:60]}...")
