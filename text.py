import os
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI
st.title("Text QA using RAG + LLaMA2")

# Text input method
input_method = st.radio("Choose input method:", ("Paste Text", "Upload .txt File"))

# Text container
text = ""

# Handle pasted text
if input_method == "Paste Text":
    text = st.text_area("Paste your text here")

# Handle text file upload
elif input_method == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

# Input for question
query = st.text_input("Ask a question about the text")

# RAG Pipeline
if text.strip() and query:
    try:
        # Convert to LangChain Document
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
        st.error(f"Error processing text: {e}")
