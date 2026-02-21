import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect, DetectorFactory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from pinecone import Pinecone

# Standardize language detection
DetectorFactory.seed = 0
app = FastAPI(title="CyberAI Pinecone Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. SETUP & EMBEDDINGS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Pinecone and VectorStore
pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings, 
    pinecone_api_key=PINECONE_API_KEY
)

# 2. MODEL SETUP
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0) 

chat_histories = {}

LANG_MAP = {
    "en": "ENGLISH", "es": "SPANISH", "fr": "FRENCH", "de": "GERMAN",
    "it": "ITALIAN", "pt": "PORTUGUESE", "zh-cn": "CHINESE", "ja": "JAPANESE", 
    "ko": "KOREAN", "ru": "RUSSIAN", "ar": "ARABIC", "hi": "HINDI"
}

# 3. RAG CHAIN SETUP
system_prompt = (
    "You are a strict Cyber Security Expert. "
    "Use ONLY the following pieces of retrieved context to answer the question: \n\n"
    "{context}\n\n"
    "RULES:\n"
    "1. If the answer is NOT in the context, say 'I am sorry, but my database does not contain information on that specific topic.'\n"
    "2. Do NOT use outside knowledge.\n"
    "3. You MUST respond in the same language as the user query.\n"
    "4. Keep the response technical and professional."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

class ChatInput(BaseModel):
    message: str
    session_id: str = "default_user"

# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_data():
    """Run this once to upload your JSON to Pinecone"""
    if os.path.exists('cyber_security.json'):
        with open('cyber_security.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        docs = [Document(page_content=entry["text"]) for entry in data if "text" in entry]
        vectorstore.add_documents(docs)
        return {"status": f"Uploaded {len(docs)} documents to Pinecone cloud!"}
    return {"error": "cyber_security.json not found"}

@app.post("/chat")
async def chat(input: ChatInput):
    try:
        raw_lang = detect(input.message)
        full_lang = LANG_MAP.get(raw_lang, "ENGLISH")
    except:
        full_lang = "ENGLISH"

    if input.session_id not in chat_histories:
        chat_histories[input.session_id] = []
    
    history = chat_histories[input.session_id][-5:]

    response = rag_chain.invoke({"input": input.message, "chat_history": history})
    answer = response["answer"]

    chat_histories[input.session_id].append(HumanMessage(content=input.message))
    chat_histories[input.session_id].append(AIMessage(content=answer))

    return {"reply": answer, "language": full_lang}