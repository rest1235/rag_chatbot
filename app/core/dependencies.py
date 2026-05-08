
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

from app.core.state import state


def get_embeddings():
    if state.embeddings is None:
        print("ВНИМАНИЕ: embeddings client не инициализирован!")
    return state.embeddings

def get_qdrant():
    if state.qdrant is None:
        print("ВНИМАНИЕ: Qdrant client не инициализирован!")
    return state.qdrant

# Вместо простого split — добавляем заголовок/контекст предыдущего чанка
def split_with_context(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    
    # Добавляем контекст предыдущего чанка к текущему
    enriched = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Берём последние 150 символов предыдущего чанка
            prev_context = chunks[i-1].page_content[-250:]
            chunk.page_content = f"[Контекст: {prev_context}]\n{chunk.page_content}"
        enriched.append(chunk)
    return enriched