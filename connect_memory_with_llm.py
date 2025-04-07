import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load Hugging Face Token
HF_TOKEN = os.environ.get("HF_TOKEN")  # Ensure this environment variable is set
if not HF_TOKEN:
    raise ValueError("Hugging Face API token is missing! Set HF_TOKEN as an environment variable.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Setup LLM (Mistral with Hugging Face)
def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        token=HF_TOKEN,  
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        task="text-generation",
        model_kwargs={ 
            "max_length": 256  
        }
    )

# Step 2: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
        You are an expert in Soybean crop cultivation and provide **accurate, structured, and concise** answers.  
        You can respond in **both Hindi and English** based on the user's question.  

        ## **User Question**:
        {question}

        ## **Context (for reference)**:
        {context}

        ## **Instructions for Response**:
        - **Use bullet points** for listing key factors.
        - **Provide concise explanations** without unnecessary details.
        - **Avoid repetition** of content.
        - If the question is **not related to soybeans**, politely decline to answer.
        - **If the question is in Hindi, answer in Hindi. If in English, answer in English.**  

        ---
        **🌱 Example Response Format (English & Hindi)**
        
        **✅ Soybean Growth Improvement Methods**  
        1️⃣ **Soil Preparation** 🏡  
        - Ensure well-drained, fertile soil rich in organic matter.  
        - Proper plowing and leveling improve soil aeration.  
        
        2️⃣ **Seed Selection** 🌱  
        - Choose disease-resistant, high-yielding soybean varieties.  
        - Certified seeds improve germination rates.  
        
        3️⃣ **Water Management** 💧  
        - Maintain **optimum soil moisture** without over-irrigation.  
        - Implement **drip irrigation** for better water utilization.  

        ---
        **🌱 सोयाबीन उत्पादन सुधारने के तरीके**  
        1️⃣ **मिट्टी की तैयारी** 🏡  
        - अच्छी जल निकासी और जैविक तत्वों से भरपूर मिट्टी का चयन करें।  
        - समुचित जुताई और समतलकरण से मिट्टी की हवा संचार में सुधार होता है।  

        2️⃣ **बीज चयन** 🌱  
        - रोग प्रतिरोधक और अधिक उत्पादन देने वाली किस्में चुनें।  
        - प्रमाणित बीज बेहतर अंकुरण दर प्रदान करते हैं।  

        3️⃣ **सिंचाई प्रबंधन** 💧  
        - मिट्टी में **संतुलित नमी** बनाए रखें, अधिक जल न डालें।  
        - **ड्रिप सिंचाई प्रणाली** का उपयोग पानी की बचत के लिए करें।  

        ---
        **Now, provide your response below following this structure.** 
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS Database
DB_FAISS_PATH = r"C:\Users\anany\Desktop\CHATBOT\vectorstore\db_faiss"

# ✅ **Fix: Use the correct way to load the embedding model**
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print("Error loading embedding model:", str(e))
    raise

# ✅ **Fix: Ensure FAISS is loaded properly**
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print("Error loading FAISS database:", str(e))
    raise

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Get User Query
user_query = input("Write Query Here: ")

# Step 6: Get Response
try:
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response.get("result", "No result found."))
    print("SOURCE DOCUMENTS: ", response.get("source_documents", []))
except Exception as e:
    print("Error during query processing:", str(e))

