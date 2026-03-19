from typing import List

from langchain.agents import create_agent
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_community.utilities import WikipediaAPIWrapper

from src.state.rag_state import RAGState


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            answer=state.answer,
        )

    def _format_docs(self, docs: List[Document], max_docs: int = 6) -> str:
        if not docs:
            return "No relevant documents found."

        parts = []
        for i, d in enumerate(docs[:max_docs], start=1):
            meta = d.metadata or {}
            source = meta.get("source") or meta.get("title") or f"doc_{i}"
            content = d.page_content.strip()
            parts.append(f"[Source {i}: {source}]\n{content}")
        return "\n\n".join(parts)

    def _build_agent(self):
        retriever = self.retriever
        format_docs = self._format_docs
        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")

        @tool
        def retriever_tool(query: str) -> str:
            """Search the user's documents, including website content, publications, resume/CV, and research background."""
            docs = retriever.invoke(query)
            return format_docs(docs)

        @tool
        def wikipedia_tool(query: str) -> str:
            """Search Wikipedia for general background information."""
            return wiki_api.run(query)

        self._agent = create_agent(
            model=self.llm,
            tools=[retriever_tool, wikipedia_tool],
            system_prompt=(
                "You are a retrieval-augmented assistant (RAG).\n\n"
                "RULES:\n"
                "1. Use retriever_tool first for any question related to:\n"
                "   - the user's website\n"
                "   - publications\n"
                "   - resume / CV\n"
                "   - research or professional background\n\n"
                "2. Base your answer on retrieved content whenever available.\n"
                "   Do not invent information.\n\n"
                "3. Use wikipedia_tool only for general knowledge questions\n"
                "   that are not answered by the user's documents.\n\n"
                "4. If the user's documents do not contain the answer, say clearly:\n"
                "   'I could not find this information in the provided documents.'\n\n"
                "5. When answering:\n"
                "   - be concise and clear\n"
                "   - summarize retrieved content\n"
                "   - avoid copying long raw passages\n\n"
                "6. For publications:\n"
                "   - summarize key publications\n"
                "   - group by year when possible\n\n"
                "7. For research or profile questions:\n"
                "   - use resume or website content\n"
                "   - provide a professional summary\n"
            ),
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        context = self._format_docs(state.retrieved_docs or [])

        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Use the provided retrieved context first. "
                            "Only call tools if the context is insufficient."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state.question}\n\n"
                            f"Retrieved context:\n{context}"
                        ),
                    },
                ]
            }
        )

        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "Could not generate answer."

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )
