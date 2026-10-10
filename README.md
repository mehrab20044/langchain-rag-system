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


## Day 29 — RAG + FastAPI + Streaming

- Added `/rag` endpoint.
- Added `/rag/stream` with streaming responses.
- Preserved context and source ID.
- Added input validation for empty queries.


## Day 30 — RAG Evaluation

Run:
PYTHONPATH=src python evaluation/evaluate.py

Dataset: 4 answerable questions and 1 unanswerable question.
Detailed results: evaluation/results.json

Results:
- Mean Precision@1: 1.0
- Mean Recall@1: 1.0
- Mean sentence BLEU on answerable questions: 82.31
- The model correctly declined to answer the unsupported version question.

Manual review:
All five answers matched the expected behavior and were supported
by the supplied context, or appropriately acknowledged missing information.

Limitations:
- The dataset is small and covers only four knowledge chunks.
- Each answerable question has one relevant chunk, so Precision@1
  and Recall@1 are equal.
- BLEU measures textual overlap, not factual correctness or groundedness.
- Correct answers with different wording can receive lower BLEU scores.
- These results do not establish performance on broader questions.