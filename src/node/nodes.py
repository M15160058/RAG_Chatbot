"""Vector store module for document embedding and retrieval"""

from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import UUID


class VectorStore:
    """Manages vector store operations"""

    def __init__(self, index_path: str = "faiss_index"):
        """
        Initialize vector store with OpenAI embeddings

        Args:
            index_path: Path to save/load FAISS index
        """
        self.embedding = OpenAIEmbeddings()
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.index_path = index_path

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents
        """
        print("Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(documents, self.embedding)

        # Default retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # Save index
        self.vectorstore.save_local(self.index_path)
        print(f"Vector store saved at {self.index_path}")

    def load_vectorstore(self):
        """Load existing FAISS index from disk"""
        print("Loading FAISS vector store...")
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embedding,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    def get_retriever(self):
        """Get retriever instance"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        return retriever.invoke(query)