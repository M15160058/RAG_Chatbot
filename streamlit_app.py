"""Streamlit UI for Agentic RAG System."""

import sys
import time
import traceback
import uuid  # <--- ADD THIS HERE
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


st.set_page_config(
    page_title="Arif AI Assistant",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": "AI-powered RAG chatbot built by Arif"
    }
)

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    """Initialize the RAG system."""
    try:
        # Validate config and initialize LLM
        llm = Config.get_llm()

        # Build document processor
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        # Process files from data folder
        documents = doc_processor.process_data_folder("data")

        if not documents:
            raise ValueError("No documents were loaded from the data folder.")

        # Build vector store
        vector_store = VectorStore()
        vector_store.create_vectorstore(documents)

        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }, 0


def main():
    """Main application."""
    init_session_state()

    st.title("🧠 Arif AI Assistant")
    st.write("Ask questions about my CV, publications, and website")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()

            if isinstance(rag_system, dict) and "error" in rag_system:
                st.error(f"Failed to initialize: {rag_system['error']}")
                st.code(rag_system["traceback"])
            else:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready. Loaded {num_chunks} document chunks.")

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
        )
        submit = st.form_submit_button("🔍 Search")

    if submit and question:
        if not st.session_state.rag_system:
            st.error("RAG system is not initialized.")
        else:
            with st.spinner("Searching..."):
                start_time = time.time()

                try:
                    result = st.session_state.rag_system.run(question)
                    elapsed_time = time.time() - start_time

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": result["answer"],
                            "time": elapsed_time,
                        }
                    )

                    st.markdown("### Answer")
                    st.success(result["answer"])

                    with st.expander("Source Documents"):
                        for i, doc in enumerate(result.get("retrieved_docs", []), start=1):
                            source = doc.metadata.get("source", f"Document {i}")
                            st.markdown(f"**{i}. {source}**")
                            st.text_area(
                                f"doc_{i}",
                                doc.page_content[:500],
                                height=140,
                                disabled=True,
                                label_visibility="collapsed",
                            )

                    st.caption(f"Response time: {elapsed_time:.2f} seconds")

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.code(traceback.format_exc())

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.caption(f"Time: {item['time']:.2f} seconds")
            st.markdown("")


if __name__ == "__main__":
    main()