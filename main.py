from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
import requests
import json
import pymongo
import faiss
import numpy as np
import pandas as pd


# Create FastAPI instance
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Embedding model
embedding_model = SentenceTransformer("thenlper/gte-large")

# Database auth

"""Establish connection to the MongoDB."""
mongo_uri = "mongodb://rwuser:Almokhtar911!@166.108.200.235:8635/test?authSource=admin&replicaSet=replica"
try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")

except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")

# Ingest data into MongoDB
db = client["satellite_data"]
collection = db["usa_states_ndvi"]

# Retrieve embeddings and IDs from MongoDB
cursor = collection.find({}, {"_id": 1})
ids = []
for document in cursor:
    ids.append(document["_id"])
#now we will use faiss to help us in vector search
index = faiss.read_index("state_ndvi_faiss_index.index")
#now we will retrive relevant data
k=10


# Ollama API config
OLLAMA_URL = 'http://166.108.224.224:3000/ollama/api/generate'
OLLAMA_TOKEN = "sk-0951c593ceee417c94e47cb554a4a976"
OLLAMA_HEADERS = {'Authorization': f'Bearer {OLLAMA_TOKEN}', 'Content-Type': 'application/json'}


def vector_search(query_embedding, index, ids, collection, num_neighbors=10):
    # Convert query embedding to the correct format for Faiss
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_vector, num_neighbors)
    
    # Retrieve the corresponding MongoDB document IDs
    nearest_ids = [ids[idx] for idx in indices[0]]
    
    # Fetch documents from MongoDB based on these IDs
    results = list(collection.find({"_id": {"$in": nearest_ids}}, {
        "state": 1,
        "mean_ndvi": 1,
        "text": 1,
        "embeddings": 1
    }))
    
    # Add distances to each document
    for i, result in enumerate(results):
        result["__nn_distance"] = distances[0][i]
    
    return results


# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str

# Helper function to fetch search results from vector database
def get_search_result(query: str) -> str:
    query_embedding = [embedding_model.encode(query)]
    
    try:
        # Get the vector search results
        search_results = vector_search(query_embedding, index, ids, collection)
        # Convert the results to a DataFrame
        df_results = pd.DataFrame(search_results)
        # Arrange columns in the desired order
        df_results = df_results[["__nn_distance", "state", "mean_ndvi", "text", "embeddings"]]
        
        # Initialize a string to store the formatted results
        search_result = ""
        # Iterate over each row in the DataFrame and format only the text output
        for _, row in df_results.iterrows():
            search_result += f"Text: {row.get('text', 'N/A')}\n"
            
        
        return (
            f"Query: {query}\nContinue to answer the query by using the Search Results:\n{search_result}."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching search results: {str(e)}")

# Helper function to stream response from Ollama API
def stream_ollama_response(query: str):
    payload = {'model': "llama3.1:8b-text-q8_0", 'prompt': query}
    with requests.post(OLLAMA_URL, headers=OLLAMA_HEADERS, json=payload, stream=True) as response:
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        for line in response.iter_lines():
            if line:  # Skip empty lines
                parsed_line = json.loads(line)
                word = parsed_line.get("response", "")
                yield word

# FastAPI route
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # Fetch search results
        enriched_query = get_search_result(request.query)
        
        # Stream response from Ollama API
        return StreamingResponse(stream_ollama_response(enriched_query), media_type="text/plain")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
