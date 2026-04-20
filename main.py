from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. INITIALIZE CLIENTS
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME"))

# Using a known stable Chat model ID
# Zephyr is excellent for chat-based RAG and very stable on the Free API
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta" 


client_HF = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def med_chat(request: ChatRequest):
    # 2. RETRIEVAL: Turn question into numbers (Embeddings)
    # This task is 'feature_extraction'
    user_vector = client_HF.feature_extraction(
        request.question, 
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. SEARCH: Query Pinecone
    search_results = index.query(
        vector=user_vector.tolist(),
        top_k=3,
        include_metadata=True
    )

    # 4. AUGMENT: Create the prompt with context
    context = "\n".join([res['metadata']['text'] for res in search_results['matches']])

    system_prompt = (
        "You are a professional medical assistant. Use the following context to answer. "
        "If you cannot find the answer in the context, say you don't know based on records. "
        "Always advise consulting a doctor."
    )
    
    # 5. GENERATE: Using the Chat Completions API
    # Note: We updated the model ID to one that explicitly supports Chat
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.question}"}
        ],
        max_tokens=400,
        temperature=0.5 # Keeps the answer focused and professional
    )

    return {
        "answer": response.choices[0].message.content, 
        "source_context": context
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)