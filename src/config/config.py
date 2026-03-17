"""Configuration module for Agentic RAG system."""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env
load_dotenv()


class Config:
    """Configuration class for RAG system."""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    USER_AGENT = os.getenv("USER_AGENT", "rag-chatbot/1.0")

    # Model Configuration
    LLM_MODEL = "openai:gpt-4o-mini"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        os.environ["USER_AGENT"] = cls.USER_AGENT

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model."""
        cls.validate()
        return init_chat_model(cls.LLM_MODEL)