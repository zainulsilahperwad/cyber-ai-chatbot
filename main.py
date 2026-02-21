import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="CyberAI Fixed")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 1. EMBEDDINGS & STORE
embeddings = PineconeEmbeddings(model="multilingual-e5-large", pinecone_api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
retriever = vectorstore.as_retriever()

# 2. MODEL
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

# 3. MODERN LCEL CHAIN (No 'langchain.chains' needed)
template = """You are a Cyber Security Expert. Use the context to answer.
Context: {context}
Question: {question}
Answer in the user's language."""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This is the "Modern Chain" that replaces the broken one
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatInput(BaseModel):
    message: str

@app.get("/")
def home(): return {"status": "online"}

@app.post("/ingest")
async def ingest():
    file_path = os.path.join(os.path.dirname(__file__), 'cyber_security.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    docs = [Document(page_content=d["text"]) for d in data]
    vectorstore.add_documents(docs)
    return {"status": "Complete"}

@app.post("/chat")
async def chat(input: ChatInput):
    # Using the new LCEL syntax
    response = rag_chain.invoke(input.message)
    return {"reply": response}