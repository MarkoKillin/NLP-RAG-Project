import requests

class OllamaLLMClient:
    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, stream: bool = False) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running and the server is accessible."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")

