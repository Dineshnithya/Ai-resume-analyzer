# rag.py

import os
import openai
from retrieve import retrieve

# Load OpenAI key from environment (.env recommended, see troubleshooting)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Threshold for relevance (tuned on example distances)
DIST_THRESHOLD = 1.0

def answer_with_context(question: str, context: list[str]) -> str:
    """
    Format the prompt with retrieved context and get answer from the LLM.
    """
    system_prompt = "You are an AI assistant. Use the provided context to answer the question."
    user_prompt = f"Context:\n\"\"\"\n{context}\n\"\"\"\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def answer_without_context(question: str) -> str:
    """
    Ask the LLM to answer the question without any retrieved context.
    """
    system_prompt = "You are an AI assistant. Answer the question to the best of your general knowledge."
    user_prompt = f"Question: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def main():
    # Example questions (in practice, get this from user/input)
    for question in ["What is React?", "What is FastAPI?"]:
        hits = retrieve(question, top_k=2)
        if not hits:
            print("No documents found in vector store.")
            continue

        top_doc, top_dist = hits[0]
        print(f"\nQuery: {question}\nClosest distance: {top_dist:.3f}")
        
        if top_dist < DIST_THRESHOLD:
            # Use RAG: incorporate context
            context_text = "\n---\n".join(doc for doc, _ in hits)
            answer = answer_with_context(question, context_text)
            print("Using RAG with context. Answer:")
        else:
            # Fallback to LLM only
            answer = answer_without_context(question)
            print("No relevant context found; using LLM directly. Answer:")

        print(answer)

if __name__ == "__main__":
    main()
