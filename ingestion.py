import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
import time
from dotenv import load_dotenv
import os

load_dotenv()

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
BATCH_SIZE = 50  # Smaller batches to stay safe on free tier

# 1. Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = InferenceClient(api_key=HF_TOKEN)

# 2. Ensure Index Exists (Interviewers love this automation)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# 3. Stream Data in Batches (Memory Efficient)
def ingest_data(file_path, num_rows=1000):
    print(f"Loading {num_rows} rows from {file_path}...")
    # Use chunksize to keep RAM usage near zero
    chunks = pd.read_csv(file_path, nrows=num_rows, chunksize=BATCH_SIZE)

    for chunk_id, df_chunk in enumerate(chunks):
        batch = []
        
        # Prepare the texts for embedding
        # We combine Description and Doctor answer for better context
        texts = [f"Question: {row['Description']}\nAnswer: {row['Doctor']}" 
                 for _, row in df_chunk.iterrows()]

        try:
            # Multi-vector embedding (more efficient than 1 by 1)
            embeddings = client.feature_extraction(
                texts, 
                model="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Build the Pinecone upload format
            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                global_idx = (chunk_id * BATCH_SIZE) + i
                batch.append({
                    "id": f"med-{global_idx}",
                    "values": vector,
                    "metadata": {"text": text[:1000]} # Limit metadata size to save space
                })

            # Upsert the batch
            index.upsert(vectors=batch)
            print(f"Successfully uploaded batch {chunk_id + 1} ({len(batch)} rows)")
            
            # Rate limiting safety for Free Tier
            time.sleep(2) 

        except Exception as e:
            print(f"Error in batch {chunk_id}: {e}")
            continue

if __name__ == "__main__":
    ingest_data("ai-medical-chatbot.csv", num_rows=1000)
    print("Ingestion Complete!")