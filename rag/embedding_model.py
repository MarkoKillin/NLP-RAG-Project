import requests
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://ollama:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimension = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for txt in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": txt},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data["embedding"]
            embeddings.append(vec)

        arr = np.array(embeddings, dtype="float32")

        if self.dimension is None:
            self.dimension = arr.shape[1]

        return arr
