import requests
import numpy as np

from rag.config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimension = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        arr = np.array(data["embeddings"], dtype="float32")

        if self.dimension is None:
            self.dimension = arr.shape[1]

        return arr
