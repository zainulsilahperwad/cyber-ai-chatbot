import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pinecone import Pinecone

app = FastAPI(title="CyberAI Direct")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- CLIENT SETUP ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

class ChatInput(BaseModel):
    message: str

@app.get("/")
def home(): return {"status": "online"}

@app.post("/chat")
async def chat(input: ChatInput):
    # 1. Get Embeddings (Using Pinecone's direct inference)
    # This replaces the entire LangChain embedding mess
    res = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[input.message],
        parameters={"input_type": "query"}
    )
    query_vector = res[0].values

    # 2. Search Pinecone
    search_res = index.query(vector=query_vector, top_k=3, include_metadata=True)
    context = "\n".join([item.metadata['text'] for item in search_res.matches if 'text' in item.metadata])

    # 3. Direct Groq Call
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"You are a Cyber Security Expert. Use this context: {context}"},
            {"role": "user", "content": input.message}
        ],
        model="llama-3.1-8b-instant",
    )
    
    return {"reply": chat_completion.choices[0].message.content}

@app.post("/ingest")
async def ingest():
    with open('cyber_security.json', 'r') as f:
        data = json.load(f)
    
    for i, entry in enumerate(data):
        # Embed each text entry
        res = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[entry["text"]],
            parameters={"input_type": "passage"}
        )
        # Upload to Pinecone
        index.upsert(vectors=[{
            "id": f"vec_{i}", 
            "values": res[0].values, 
            "metadata": {"text": entry["text"]}
        }])
        
    return {"status": "Ingest complete"}