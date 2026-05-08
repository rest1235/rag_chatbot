from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


class AppState:
    def __init__(self):
        self.embeddings: HuggingFaceEmbeddings = None
        self.qdrant: QdrantClient = None

state = AppState()