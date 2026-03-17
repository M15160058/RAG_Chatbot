"""State definition for RAG workflow."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import uuid  # <--- Add this line
from uuid import UUID


class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: Optional[str] = None