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


## Day 27 — Simple RAG

Built a simple grounded RAG pipeline:

Query → Retrieval → Context → LLM → Answer + Sources

- Retrieved the most relevant chunk using embeddings and cosine similarity.
- Built a grounded prompt from retrieved context.
- Generated answers with ToGPT.
- Returned sources with the answer.
- Tested out-of-context questions successfully.