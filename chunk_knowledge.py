# chunk_knowledge.py

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split the text into chunks of `chunk_size` words, with `overlap` words overlap.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    if not words:
        return chunks
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move by non-overlapping portion
    return chunks

# Example usage:
# text = "..."  # long document
# chunks = chunk_text(text, chunk_size=200, overlap=50)
# print(len(chunks), "chunks created.")
