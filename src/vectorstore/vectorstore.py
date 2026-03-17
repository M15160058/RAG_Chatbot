"""Vector store module for document embedding and retrieval."""

from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStore:
    """Manages vector store operations."""

    def __init__(self, index_path: str = "faiss_index"):
        self.embedding = OpenAIEmbeddings()
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.index_path = index_path

    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents."""
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        """Return retriever."""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant docs."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)