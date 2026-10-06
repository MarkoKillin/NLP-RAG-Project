from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    index_dir: Path = BASE_DIR / "index"

    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    top_k: int = Field(default=5, ge=1)

    bm25_stem: bool = True
    bm25_remove_stopwords: bool = False

    rrf_k: int = Field(default=60, gt=0)
    hybrid_candidates: int = Field(default=50, ge=1)

    embedding_model_name: str = "nomic-embed-text"

    ollama_base_url: str = "http://ollama:11434"
    ollama_model_name: str = "llama3.2:3b-instruct-q8_0"

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def _overlap_below_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


settings = Settings()
