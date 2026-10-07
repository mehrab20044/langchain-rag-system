from pathlib import Path
import os

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def load_chunks() -> list[str]:
    text = Path("knowledge.txt").read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


def embed_chunks(chunks: list[str]):
    return model.encode(chunks)


def embed_query(query: str):
    return model.encode(query)


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

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

    query = "What is Pinecone used for?"
    query_embedding = embed_query(query)

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
                "source": "knowledge.txt",
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