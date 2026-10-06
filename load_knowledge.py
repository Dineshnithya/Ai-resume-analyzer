# load_knowledge.py
import os

def load_documents(knowledge_dir: str = "knowledge") -> list[str]:
    """
    Load all text documents from the given knowledge directory.
    Returns a list of document strings.
    """
    texts = []
    for fname in os.listdir(knowledge_dir):
        path = os.path.join(knowledge_dir, fname)
        if os.path.isfile(path) and fname.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

# Example usage:
# docs = load_documents("knowledge")
# print(f"Loaded {len(docs)} knowledge documents.")
