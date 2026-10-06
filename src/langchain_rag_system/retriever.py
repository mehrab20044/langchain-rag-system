from pathlib import Path
import numpy as np
import os 
import requests
from dotenv import load_dotenv



from sentence_transformers import SentenceTransformer

load_dotenv()

TOGPT_API_KEY = os.getenv("TOGPT_API_KEY")
TOGPT_URL = "https://togpt.ir/api/v1/chat/completions"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

def load_chunks() -> list[str]:
    text = Path("knowledge.txt").read_text(encoding="utf-8")

    chunks =[
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks

def embed_chunks(chunks: list[str]):
    return model.encode(chunks)

def embed_query(query:str):
    return model.encode(query)

def cosine_similarity(vector_a,vector_b):
    dot_product = np.dot(vector_a,vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return dot_product / (norm_a * norm_b)



def build_prompt(query: str, context: str) -> str:
    return f"""
You are a grounded assistant.

Answer the question using only the provided context.

If the answer is not in the context, say:
"I don't have enough information to answer."

Context:
{context}

Question:
{query}
"""

if __name__ == "__main__":
    chunks = load_chunks()

    chunk_embeddings = embed_chunks(chunks)
    query_embedding = embed_query("What is Pinecone used for?")



    print("Chunks:", len(chunks))
    print("Chunk embeddings shape:", chunk_embeddings.shape)
    print("Query embedding shape:", query_embedding.shape)
    
    
    results = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        results.append(
            {
                "text": chunks[index],
                "score": float(score),
                "source": "knowledge.txt"
            }
        )

    results.sort(
    key=lambda item: item["score"],
    reverse=True,
)

top_result = results[0]

context = top_result["text"]

print("Context:")
print(context)

print("Top result:")
print(top_result)


query = "What is the capital of France?"

prompt = build_prompt(
    query,
    context,
)

print("Prompt:")
print(prompt)


def generate_answer(prompt: str) -> str:
    response = requests.post(
        TOGPT_URL,
        headers={
            "Authorization": f"Bearer {TOGPT_API_KEY}",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]

answer = generate_answer(prompt)

print("Answer:")
print(answer)

if answer == "I don't have enough information to answer.":
    final_response = {
        "answer": answer,
        "sources": [],
    }
else:
    final_response = {
        "answer": answer,
        "sources": [top_result["source"]],
    }