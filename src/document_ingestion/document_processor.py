"""Document processing module for loading and splitting documents."""

from typing import List, Union
from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_url_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load URLs from a text file and fetch their contents."""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            urls = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        docs: List[Document] = []
        for url in urls:
            try:
                docs.extend(self.load_from_url(url))
                print(f"Loaded URL: {url}")
            except Exception as e:
                print(f"Skipped URL {url}: {e}")

        return docs

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a single PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_docx(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a DOCX file."""
        loader = UnstructuredWordDocumentLoader(str(file_path))
        return loader.load()

    def load_from_csv(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a CSV file."""
        loader = CSVLoader(str(file_path))
        return loader.load()

    def load_from_md(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a Markdown file."""
        loader = UnstructuredMarkdownLoader(str(file_path))
        return loader.load()

    def load_from_html(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from an HTML file."""
        loader = UnstructuredHTMLLoader(str(file_path))
        return loader.load()

    def load_from_json(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a JSON file."""
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = json.dumps(data, indent=2, ensure_ascii=False)
        return [
            Document(
                page_content=text,
                metadata={"source": str(file_path), "file_type": "json"},
            )
        ]

    def load_single_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single file based on extension."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        # Special handling for url.txt
        if file_path.name.lower() == "url.txt":
            return self.load_from_url_file(file_path)

        if suffix == ".pdf":
            return self.load_from_pdf(file_path)
        elif suffix == ".txt":
            return self.load_from_txt(file_path)
        elif suffix == ".docx":
            return self.load_from_docx(file_path)
        elif suffix == ".csv":
            return self.load_from_csv(file_path)
        elif suffix == ".md":
            return self.load_from_md(file_path)
        elif suffix in [".html", ".htm"]:
            return self.load_from_html(file_path)
        elif suffix == ".json":
            return self.load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.name}")

    def load_from_data_folder(self, folder: Union[str, Path] = "data") -> List[Document]:
        """Load all supported documents from the data folder."""
        docs: List[Document] = []
        folder_path = Path(folder)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if not folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        for file_path in folder_path.iterdir():
            if file_path.is_file():
                try:
                    docs.extend(self.load_single_file(file_path))
                    print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Skipped {file_path.name}: {e}")

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)

    def process_data_folder(self, folder: Union[str, Path] = "data") -> List[Document]:
        """Load and split all supported documents from the data folder."""
        docs = self.load_from_data_folder(folder)
        return self.split_documents(docs)