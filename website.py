import os
import requests
from bs4 import BeautifulSoup

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Streamlit UI
st.title("Website QA using RAG + LLaMA2")

# Input for website URL
url = st.text_input("Enter Website URL")

# Input for question
query = st.text_input("Ask a question about the website")

# RAG Pipeline
if url and query:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=60)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        if len(text.strip()) == 0:
            st.error("The website doesn't contain extractable text.")
        else:
            documents = [Document(page_content=text)]

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

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

            with st.expander("Sources"):
                for doc in result.get("source_documents", []):
                    st.markdown(doc.page_content[:500] + "...")

    except Exception as e:
        st.error(f"Error fetching website: {e}")
