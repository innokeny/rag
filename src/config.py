from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    docs_dir: str
    index_dir: str
    chunks_dir: str
    benchmark_file: str

    embedding_model: str
    llm_model: str

    chunk_size: int
    chunk_overlap: int

    top_k: int
    similarity_threshold: float

    max_new_tokens: int
    temperature: float

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

config = Config() # pyright: ignore[reportCallIssue]
