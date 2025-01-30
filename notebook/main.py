from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import load_index_from_storage

# Set up embedding model and index
embedding_model_name = '../../bge-m3'
embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, max_length=1024, device='cpu')

Settings.llm = None
Settings.embed_model = embed_model

persist_dir = "../../index"
vector_store = FaissVectorStore.from_persist_dir(persist_dir)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
index = load_index_from_storage(storage_context=storage_context)

query_engine = index.as_query_engine(similarity_top_k=5)

# Define FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_endpoint(request: QueryRequest):
    query = request.query
    result = query_engine.query(query)
    # Assuming result can be converted to a JSON serializable format
    return {"result": result}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
