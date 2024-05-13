import asyncio
from typing import Any
from db.vector import VectorDB
from external.vectorizer import Vectorizer
from tqdm.asyncio import tqdm_asyncio
from config import (
    CHROMA_CLIENT,
    RAG_TOP_K,
    VECTORIZER_HOST,
    VECTORIZER_PORT
)


class ChromaDB(VectorDB):
    def __init__(
        self,
        n_parallelism: int = 1000,
    ) -> None:

        super().__init__()
        self.client = CHROMA_CLIENT
        self.n_parallelism = n_parallelism

    @staticmethod
    def _to_batch(documents: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
        batches = []
        for i in range(0, len(documents), batch_size):
            _ids = []
            _text = []
            _meta_data = []
            documents_batch = documents[i : i + batch_size]
            for doc in documents_batch:
                _ids.append(doc["id"])
                _text.append(doc["text"])
                _meta_data.append(doc["meta_data"])
            batches.append({"id": _ids, "text": _text, "meta_data": _meta_data})
        return batches

    async def insert(
        self, documents: list[dict[str, Any]], collection_name: str, batch_size: int
    ) -> None:
        collection = self.client.get_or_create_collection(name=collection_name)
        batches = self._to_batch(documents, batch_size=self.n_parallelism)
        async with Vectorizer(
            f"{VECTORIZER_HOST}:{VECTORIZER_PORT}", batch_size=batch_size
        ) as vectorizer:
            batches_embeddings = await tqdm_asyncio.gather(
                *[vectorizer.vectorize(batch["text"]) for batch in batches]
            )

        for i, batch in enumerate(batches):
            collection.upsert(
                        ids=batch["id"],
                        embeddings=batches_embeddings[i],
                        metadatas=batch["meta_data"],
                        documents=batch["text"],
            )  

    async def delete(self, collection_name: str, id: list[str]) -> None:
        collection = self.client.get_or_create_collection(name=collection_name)
        asyncio.create_task(
            ChromaDB.run_async(
                collection.delete,
                ids=id,
            )
        )

    async def get(self, collection_name: str, id: list[str]) -> list[dict[str, Any]]:
        collection = self.client.get_or_create_collection(name=collection_name)
        result = await ChromaDB.run_async(
            collection.get,
            ids=id,
        )
        return [
            {
                "id": _id,
                "text": text,
                "meta_data": meta_data,
            }
            for _id, text, meta_data in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    async def search(
        self, query: str, collection_name: str, top_k: int = RAG_TOP_K
    ) -> list[dict[str, Any]]:
        collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Calling: {VECTORIZER_HOST}:{VECTORIZER_PORT}/embed")
        async with Vectorizer(f"{VECTORIZER_HOST}:{VECTORIZER_PORT}/embed") as vectorizer:
            vector = await vectorizer.vectorize([query])
        result = await ChromaDB.run_async(
            collection.query,
            query_embeddings=vector[0],
            n_results=top_k,
        )
        return [
            {
                "id": _id,
                "text": text,
                "meta_data": meta_data,
                "distance": distance,
            }
            for _id, text, meta_data, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
