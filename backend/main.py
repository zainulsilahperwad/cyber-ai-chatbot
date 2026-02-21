import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Updated imports for modern LangChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

app = FastAPI(title="CyberAI Ultra-Light")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLOUD CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 1. CLOUD EMBEDDINGS (1024 Dimensions)
embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", 
    pinecone_api_key=PINECONE_API_KEY
)

# 2. CONNECT TO PINECONE
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings, 
    pinecone_api_key=PINECONE_API_KEY
)

# 3. LLM SETUP
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

# --- RAG SETUP ---
system_prompt = (
    "You are a strict Cyber Security Expert. Use ONLY the provided context. "
    "If unknown, say you don't know. Respond in the user's language."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Create the chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- MODELS ---
class ChatInput(BaseModel):
    message: str
    session_id: str = "default"

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "CyberAI is running"}

@app.post("/ingest")
async def ingest():
    # Dynamic path fix for Render environments
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'cyber_security.json')
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    docs = [Document(page_content=d["text"]) for d in data]
    vectorstore.add_documents(docs)
    return {"status": "Cloud indexing complete"}

@app.post("/chat")
async def chat(input: ChatInput):
    # Using the defined chain
    response = chain.invoke({"input": input.message, "chat_history": []})
    return {"reply": response["answer"]}