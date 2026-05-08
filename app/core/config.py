from fastembed import SparseTextEmbedding
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_HOST: str
    APP_PORT: str
    APP_DOMEN: str
    APP_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    COLLECTION_NAME: str = "pdf_documents"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()