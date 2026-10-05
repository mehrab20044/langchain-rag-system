from fastapi import FastAPI

app = FastAPI(
    title = "LangChain RAG System",
)

@app.get("/")
def root():
    return {
        "message": "LangChain RAG System is running"
    }