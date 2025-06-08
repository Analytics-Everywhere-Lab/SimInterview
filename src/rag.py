from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from variables import GPT_EMBEDDING_MODEL
from langchain.schema import Document

import os


# === Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Directory to persist Chroma vector DBs
PERSIST_DIR = "../chroma_db"

def get_rag_configs():
    """Initialize and return RAG configurations."""
    # Embedding model for RAG
    embeddings = OpenAIEmbeddings(
        model=GPT_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    # Text splitter for chunking documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150, 
                        length_function=len, separators=["\n\n", "\n", ".", " ", ""]
                    ) 
    return embeddings, text_splitter

def save_doc_to_vector_store(text: str, collection_name: str, file_name: str = None):
    """Chunk text and index into Chroma under a specific collection."""
    embeddings, text_splitter = get_rag_configs()
    chunks = text_splitter.split_text(text)
    metadatas = []
    for i, chunk in enumerate(chunks):
        md = {
            "source": collection_name,
            "chunk_id": i,
        }
        if file_name is not None:
            md["file_name"] = file_name
        metadatas.append(md)

    docs = [
        Document(page_content=chunks[i], metadata=metadatas[i])
        for i in range(len(chunks))
    ]
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    # Add texts with metadata
    vectordb.add_documents(documents=docs)
    return vectordb
