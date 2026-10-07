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



## Day 28 — LangChain Core

Rebuilt the RAG pipeline with LangChain components:

Query → Retriever → PromptTemplate → ChatOpenAI → StrOutputParser → Answer

### Comparison

Manual version:
- more direct control
- more request/parsing code

LangChain version:
- cleaner component structure
- easier to replace models and pipeline parts
- adds extra abstraction and dependencies