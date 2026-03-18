from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

app = FastAPI(title="Arif AI Assistant API")

rag_system = None


class ChatRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    global rag_system

    llm = Config.get_llm()

    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    documents = doc_processor.process_data_folder("data")
    if not documents:
        raise ValueError("No documents found in data folder.")

    vector_store = VectorStore()
    vector_store.create_vectorstore(documents)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm,
    )
    graph_builder.build()

    rag_system = graph_builder


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    result = rag_system.run(request.question)

    return {
        "question": request.question,
        "answer": result["answer"],
        "retrieved_docs": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:300],
            }
            for doc in result.get("retrieved_docs", [])
        ],
    }
