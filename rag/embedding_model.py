import requests
import numpy as np

from rag.config import settings


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        base_url: str = settings.ollama_base_url,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

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

        return np.array(data["embeddings"], dtype="float32")
