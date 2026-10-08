from fastapi import FastAPI
from pydantic import BaseModel,Field

from langchain_rag_system.langchain_rag import (
    chain,
    retrieve_result,
    )
from fastapi.responses import StreamingResponse

app = FastAPI(
    title = "LangChain RAG System",
)
class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

@app.get("/")
def root():
    return {
        "message": "LangChain RAG System is running"
    }

@app.post("/rag")
def rag(request: QueryRequest):
    answer = chain.invoke(request.query)

    return {
        "answer": answer
    }


@app.post("/rag/stream")
def rag_stream(request: QueryRequest):
    retrieved = retrieve_result(request.query)

    def generate():
        yield f"source_id: {retrieved['source_id']}\n"
        yield f"context: {retrieved['text']}\n\n"

        for chunk in chain.stream(request.query):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )