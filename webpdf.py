import os
import tempfile
import requests
from bs4 import BeautifulSoup

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI
st.title(" Website/PDF QA with RAG + LLaMA2")

# Option for PDF or URL
option = st.radio("Choose input source:", ("Upload PDF", "Enter Website URL"))

# User question
query = st.text_input("Ask a question about the content")

# Document container
documents = []

#  PDF Upload
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

#  Website Input
elif option == "Enter Website URL":
    url = st.text_input("Enter a Website URL")
    if url:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            if len(text.strip()) == 0:
                st.error("The website doesn't contain extractable text.")
            else:
                documents = [Document(page_content=text)]

        except Exception as e:
            st.error(f"Error fetching website: {e}")

# RAG Pipeline
if documents and query:
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    if len(texts) == 0:
        st.error("Text splitting failed. No valid chunks to embed.")
    else:
        # Embedding + Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(texts, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Load Ollama LLM
        llm = OllamaLLM(model="phi")

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run QA
        result = qa_chain.invoke({"query": query})

        st.subheader("Answer:")
        st.write(result["result"])

        # Show source content (optional)
        with st.expander("Sources"):
            for doc in result.get("source_documents", []):
                st.markdown(doc.page_content[:500] + "...")

