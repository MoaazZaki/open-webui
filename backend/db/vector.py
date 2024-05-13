import abc
import asyncio
import os
from typing import Any, Final
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential

MAX_THREADS = 10


# Abstract class for vector database operations
class VectorDB(abc.ABC):
    executor: Final[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=MAX_THREADS)

    def __init__(self) -> None:
        pass
    @abc.abstractmethod
    async def insert(
        self, documents: list[dict[str, Any]], collection_name: str, batch_size: int
    ) -> None:
        pass

    @abc.abstractmethod
    async def delete(self, collection_name: str, id: list[str]) -> None:
        pass

    @abc.abstractmethod
    async def get(self, collection_name: str, id: list[str]) -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def search(self, query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
        pass

    @staticmethod
    @retry(wait=wait_random_exponential(multiplier=1, min=0.5, max=5))
    async def run_async(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(VectorDB.executor, lambda: func(*args, **kwargs))
