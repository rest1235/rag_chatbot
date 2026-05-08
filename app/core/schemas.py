from pydantic import BaseModel

class ChunkResult(BaseModel):
    text: str
    source: str
    page: int
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]

class IngestResponse(BaseModel):
    status: str
    pages: int
    chunks: int
    collection: str