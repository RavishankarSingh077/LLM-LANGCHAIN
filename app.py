from flask import Flask, request, jsonify, render_template
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set LangChain API config
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Flask app setup
app = Flask(__name__)

# LangChain components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question:{question}")
])

llm = Ollama(model="phi")  # You can change to "llama2", "mistral", etc.
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_question = request.form["question"]
        answer = chain.invoke({"question": user_question})
    return render_template("index.html", answer=answer)

# API endpoint (optional for frontend or testing)
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.json
    question = data.get("question", "")
    response = chain.invoke({"question": question})
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
