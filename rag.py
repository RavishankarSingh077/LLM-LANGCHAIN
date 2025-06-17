import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
# Set LangChain API config
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize Streamlit App
st.title("📄 Transformer Paper QA (RAG + LLAMA2)")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF (e.g., Attention Is All You Need)", type="pdf")
query = st.text_input("Ask a question about the PDF")

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load Ollama LLM


    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(model="phi")

    # Retrieval-based QA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        result = qa_chain.invoke({"query": query})
        st.subheader("Answer:")
        st.write(result["result"])

        with st.expander("Sources"):
            for doc in result.get("source_documents", []):
                st.markdown(doc.page_content[:500] + "...")

