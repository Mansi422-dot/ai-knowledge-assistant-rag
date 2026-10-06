from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "./data"
    max_upload_mb: int = 15
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    similarity_threshold: float = 0.25

settings = Settings()
