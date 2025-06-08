from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader

def load_documents():
    loader = DirectoryLoader(

import os
import gradio as gr
import fitz  # PyMuPDF for PDF parsing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# === Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Directory to persist Chroma vector DBs
PERSIST_DIR = "./chroma_db"

# Initialize embedding model and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

# Text splitter for chunking documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def extract_text_from_pdfs(files):
    """Extract full text from a list of PDF files."""
    text = ""
    for f in files or []:
        pdf = fitz.open(stream=f.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    return text

def chunk_and_index(text, collection_name):
    """Chunk text and index into Chroma under a specific collection."""
    docs = splitter.split_text(text)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    # Add texts with metadata
    vectordb.add_texts(texts=docs, metadatas=[{"source": collection_name}] * len(docs))
    vectordb.persist()

def generate_feedback(cv_text, jd_text):
    """Use LLM to generate resume improvement suggestions."""
    prompt = f"""
You are a skilled career advisor. 
Given the following Job Description and the Candidate's Resume, 
provide actionable bullet-point feedback on how to improve the resume 
to better match the job requirements.

Job Description:
\"\"\"
{jd_text}
\"\"\"

Candidate Resume:
\"\"\"
{cv_text}
\"\"\"

Please list at least 5 suggestions.
"""
    return llm(prompt)

def handle_upload(cv_files, jd_files):
    """Main handler: extract, index, and feedback."""
    if not cv_files or not jd_files:
        return "❌ Please upload both CV and Job Description PDFs."
    # Extract texts
    cv_text = extract_text_from_pdfs(cv_files)
    jd_text = extract_text_from_pdfs(jd_files)
    # Index into RAG vector DB
    chunk_and_index(cv_text, "cv_collection")
    chunk_and_index(jd_text, "jd_collection")
    # Generate feedback
    feedback = generate_feedback(cv_text, jd_text)
    return feedback

# === Gradio UI ===
with gr.Blocks(title="RAG CV-JD Analyzer") as demo:
    gr.Markdown("## Upload CV & Job Description")
    with gr.Row():
        cv_upload = gr.File(label="Candidate CV (PDF)", file_types=[".pdf"], file_count="multiple")
        jd_upload = gr.File(label="Job Description (PDF)", file_types=[".pdf"], file_count="multiple")
    analyze_btn = gr.Button("Analyze & Feedback")
    feedback_box = gr.Textbox(label="Improvement Suggestions", lines=10, interactive=False)

    analyze_btn.click(
        fn=handle_upload,
        inputs=[cv_upload, jd_upload],
        outputs=feedback_box
    )

demo.launch()
