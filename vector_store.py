# vector_store.py

import chromadb
from chromadb.config import Settings

class VectorStore:
    """
    Wrapper around ChromaDB client/collection.
    """

    def __init__(self, persist_dir="chroma_db", collection_name="knowledge"):
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings(self, texts: list[str], embeddings: list[list[float]], ids=None):
        """
        Add documents and their embeddings to the collection.
        """
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def query(self, query_embeds: list[list[float]], n_results: int = 3):
        """
        Query the collection with given query embeddings. Returns dict with 'documents' and 'distances'.
        """
        results = self.collection.query(
            query_embeddings=query_embeds,
            n_results=n_results,
            include=["documents", "distances"]
        )
        return results

    def persist(self):
        """
        Persist the database to disk (for in-memory usage).
        """
        self.client.persist()

# Example usage:
# store = VectorStore()
# store.add_embeddings(["doc1", "doc2"], [[0.1,0.2],[0.3,0.4]])
# results = store.query([[0.1,0.2]], n_results=1)
# print(results)
