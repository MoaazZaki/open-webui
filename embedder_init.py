import os
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

MODEL_NAME = os.environ.get("MODEL_NAME", 'intfloat/multilingual-e5-base')
DEVICE = os.environ.get('DEVICE', 'cpu')
BATCH_SIZE =int(os.environ.get('BATCH_SIZE', 32))
ENGINE = os.environ.get("MODEL_ENGINE", "optimum")
DTYPE = os.environ.get("DTYPE", "float16") 

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