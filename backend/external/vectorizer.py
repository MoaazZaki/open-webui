from typing import Union
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from tenacity import retry, wait_random_exponential
import asyncio

WORKERS_NUMBER = 4

class Vectorizer:
    """
    A class to vectorize text inputs using an async aiohttp wrapper to call an embedder server.
    """

    def __init__(self, server_url, batch_size=100):
        """
        Constructor for the Vectorizer class.

        Args:
            server_url (str): The URL of the embedder server.
            batch_size (int, optional): The number of inputs to send in each request. Defaults to 100.
        """
        self.server_url = server_url
        self.batch_size = batch_size
        self.session = aiohttp.ClientSession()

    async def __aenter__(self):
        """
        Enter method for async context manager.
        """
        await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit method for async context manager.
        """
        await self.session.__aexit__(exc_type, exc_val, exc_tb)

    @retry(wait=wait_random_exponential(multiplier=1, min=0.5, max=5))
    async def _post(self, payload):
        """
        Private method to send a POST request to the server.

        Args:
            payload (dict): The request payload.

        Returns:
            dict: The server's response.
        """
        async with self.session.post(self.server_url, json=payload) as response:
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                raise aiohttp.ClientResponseError(response.request_info, response.history)

    async def vectorize(self, inputs: Union[str, list[str]], normalize=False, truncate=False):
        """
        Vectorize the provided inputs.

        Args:
            inputs (list[str]): The inputs to vectorize.
            normalize (bool, optional): Whether to normalize the inputs. Defaults to False.
            truncate (bool, optional): Whether to truncate the inputs. Defaults to False.

        Returns:
            list: The vectorized inputs.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            return []

        # Split inputs into batches
        tasks = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            payload = {"inputs": batch, "normalize": normalize, "truncate": truncate}
            tasks.append(self._post(payload))

        # Gather all results
        print(f"Starting vectorizer inference with {len(tasks)} requests")
        results = []
        for i in tqdm(range(0,len(tasks),WORKERS_NUMBER)):
            results.extend(await tqdm_asyncio.gather(*tasks[i: i+WORKERS_NUMBER]))
        # [await task for task in tqdm(tasks)]
            
        #results = await tqdm_asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    async def close(self):
        """
        Close the aiohttp.ClientSession.
        """
        await self.session.close()
