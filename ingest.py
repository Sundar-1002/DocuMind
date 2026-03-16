import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma                  
from langchain_text_splitters import RecursiveCharacterTextSplitter 


load_dotenv()

DATA_PATH = "data/"
CHROMA_DB = "chroma_db/"

def ingest():
    print("Loading Documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        print("No documents found in the specified directory.")
        return
    
    print(f"Loaded {len(documents)} pages from PDF documents.")

    print("Splitting Documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Generating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings Generated.")

    Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory=CHROMA_DB
    )
    print(f"Done. {len(chunks)} chunks saved to '{CHROMA_DB}'")

if __name__ == "__main__":
    ingest()