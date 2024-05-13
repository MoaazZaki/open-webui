import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


MODEL_NAME = os.environ.get("MODEL_NAME", 'intfloat/multilingual-e5-base')
DEVICE = os.environ.get('DEVICE', 'cpu')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE','32'))
ENGINE = os.environ.get("MODEL_ENGINE", "optimum")
DTYPE = os.environ.get("DTYPE","int8") 

class Input(BaseModel):
    inputs: list[str]

app = FastAPI()

engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(
        model_name_or_path=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lengths_via_tokenize=False,
        model_warmup=False,
        engine=ENGINE,
        dtype=DTYPE
    )
)
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20, model_name=MODEL_NAME, tokens_per_chunk=200)

@app.on_event("startup")
async def startup_event():
    await engine.astart()

@app.on_event("shutdown")
async def shutdown_event():
    await engine.aclose() 
    
@app.post("/embed", response_model=list[list[float]])
async def embed(body: Input):
    return [np.mean((await engine.embed(sentences=splitter.split_text(text)))[0],axis=0) for text in body.inputs]
