# LangChain RAG System

FastAPI project for building a RAG pipeline.

## Day 26 — RAG Architecture

Flow:

Query
→ Retrieval
→ Ranking / Reranking
→ Context
→ LLM
→ Answer + Sources

Schemas:

- QueryRequest: query
- ContextChunk: text, source, score
- AnswerResponse: answer, sources

Notes:

- Retrieval finds relevant chunks.
- Generation builds the answer from context.
- Sources must be traceable.