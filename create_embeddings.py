# create_embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings
from load_knowledge import load_documents
from chunk_knowledge import chunk_text

def main():
    # 1. Load and chunk all knowledge documents
    docs = load_documents("knowledge")
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, chunk_size=200, overlap=50)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No knowledge chunks found. Make sure you have .txt files in 'knowledge/'.")
        return

    # 2. Initialize embedding model (HuggingFace SentenceTransformer)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # fast, reasonably accurate
    print(f"Embedding model loaded, encoding {len(all_chunks)} chunks...")

    # 3. Compute embeddings (in batches if large)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 4. Initialize ChromaDB client (persistent to chroma_db/ folder)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_db"))
    collection = client.get_or_create_collection("knowledge")
    
    # 5. Add embeddings and documents to Chroma
    print("Uploading embeddings to ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(all_chunks))],
        ids=[f"id_{i}" for i in range(len(all_chunks))]
    )
    client.persist()  # save to disk
    print(f"{len(all_chunks)} chunks indexed in chroma_db/.")

if __name__ == "__main__":
    main()
