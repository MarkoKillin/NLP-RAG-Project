import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag.config import settings
from rag.models import RetrievedChunkModel


class CrossEncoderReranker:
    """Re-scores retrieved chunks with a cross-encoder that reads question and chunk together."""

    def __init__(self, model_name: str = settings.reranker_model_name):
        self.model_name = model_name
        self._loaded = None

    def load(self):
        # Lazy so BM25-only runs never touch the model.
        if self._loaded is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            self._loaded = (tokenizer, model)
        return self._loaded

    @torch.inference_mode()
    def rerank(
        self, query: str, chunks: list[RetrievedChunkModel], top_k: int
    ) -> list[RetrievedChunkModel]:
        if not chunks:
            return []
        tokenizer, model = self.load()
        inputs = tokenizer(
            [query] * len(chunks),
            [c.content for c in chunks],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()
        ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:top_k]
        return [chunk.model_copy(update={"score": score}) for chunk, score in ranked]


if __name__ == "__main__":
    # The Docker entrypoint runs this to download the model before the app starts.
    CrossEncoderReranker().load()
