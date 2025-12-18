"""  
MedBRAIN AI - PDF Ingestion Script
Processes Gale Medical Encyclopedia PDF and creates vector database
"""

import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_PATH = "data/gale_medical_encyclopedia.pdf"
CHROMA_DB_PATH = "chroma_db"

def ingest_pdf(pdf_path: str = PDF_PATH, chroma_path: str = CHROMA_DB_PATH):
    """
    Ingest PDF and create Chroma vector database
    
    Args:
        pdf_path: Path to the medical encyclopedia PDF
        chroma_path: Path to store the Chroma database
    """
    
    print("📚 Starting PDF ingestion process...")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF not found at {pdf_path}")
        print("\nPlease download the PDF from:")
        print("https://huggingface.co/datasets/amalsp/MEDICAL_ENCYCLOPEDIA/blob/main/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf")
        print(f"\nAnd save it to: {pdf_path}")
        sys.exit(1)
    
    # Load PDF
    print(f"📄 Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    
    # Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Initialize embeddings model
    print("🤖 Initializing embedding model (sentence-transformers)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding model loaded")
    
    # Create vector database
    print(f"📦 Creating Chroma vector database at {chroma_path}...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    vectordb.persist()
    print(f"✅ Vector database created and persisted!")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    test_query = "What is diabetes?"
    results = vectordb.similarity_search(test_query, k=2)
    print(f"Query: {test_query}")
    print(f"Found {len(results)} relevant chunks\n")
    
    if results:
        print("Sample result:")
        print(results[0].page_content[:200] + "...")
    
    print("\n✨ Ingestion complete! MedBRAIN AI is ready to use.")
    print(f"Vector database location: {chroma_path}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Run ingestion
    ingest_pdf()
