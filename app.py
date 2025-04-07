from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv(find_dotenv())

app = Flask(__name__)

# Path to FAISS vector database
DB_FAISS_PATH = r"C:\Users\anany\Desktop\CHATBOT\vectorstore\db_faiss"

# Load FAISS vectorstore
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Define custom prompt
def set_custom_prompt():
    custom_prompt_template = """
    You are a soybean expert who answers in Hindi and English.
    Provide detailed responses. Avoid small talk.

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load Hugging Face LLM
def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        temperature=0.5,
        max_length=512,
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

# Homepage route
@app.route('/')
def home():
    return render_template("index.html")

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"response": "Please enter a question."})

    try:
        # Load vectorstore and QA chain
        vectorstore = get_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        # Get response
        response = qa_chain.invoke({'query': prompt})
        result = response["result"]
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
