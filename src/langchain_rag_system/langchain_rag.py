import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain_rag_system.retriever import (
    cosine_similarity,
    embed_chunks,
    embed_query,
    load_chunks,
)


load_dotenv()

TOGPT_API_KEY = os.getenv("TOGPT_API_KEY")

output_parser = StrOutputParser()


prompt_template = PromptTemplate.from_template(
    """
You are a grounded assistant.

Answer the question using only the provided context.

If the answer is not in the context, say:
"I don't have enough information to answer."

Context:
{context}

Question:
{query}
"""
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=TOGPT_API_KEY,
    base_url="https://togpt.ir/api/v1",
)


def retrieve_result(query: str) -> dict:
    chunks = load_chunks()

    chunk_embeddings = embed_chunks(chunks)
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
                "source_id": f"knowledge.txt#chunk-{index + 1}",
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[0]


def retrieve_context(query: str) -> str:
    result = retrieve_result(query)

    return result["text"]


retriever_component = RunnableLambda(retrieve_context)


chain = (
    {
        "context": retriever_component,
        "query": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | output_parser
)


if __name__ == "__main__":
    result = chain.invoke(
        "What is Pinecone used for?"
    )

    print(result)