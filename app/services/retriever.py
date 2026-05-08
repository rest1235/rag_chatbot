from langchain_groq import ChatGroq
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from app.core.config import settings
from app.core.dependencies import get_embeddings, get_qdrant
from app.core.schemas import AskResponse, ChunkResult
from app.services.ingest import sparse_model

def search_similar_chunks(query: str) -> list[ChunkResult]:
    embeddings_model = get_embeddings()
    
    # Dense вектор
    dense_vector = embeddings_model.embed_query(query)
    
    # Sparse вектор
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vector = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )

    client = get_qdrant()

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=10
            ),
            Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=10
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=settings.TOP_K
    ).points

    return [
        ChunkResult(
            text=r.payload["text"],
            source=r.payload["source"],
            page=r.payload["page"],
            score=r.score
        )
        for r in results
    ]

def ask_question(query: str) -> dict:
    chunks = search_similar_chunks(query)

    if not chunks:
        return AskResponse(
            answer="Не нашёл релевантной информации в документах.",
            sources=[]
        )

    # 2. Собираем контекст из чанков
    context = "\n\n---\n\n".join([c.text for c in chunks])

    # 3. Формируем промпт
    prompt = f"""Ты помощник для ответов на вопросы по документам.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа нет в контексте — так и скажи.

Контекст:
{context}

Вопрос: {query}

Ответ:"""

    # 4. Отправляем в LLM
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )
    response = llm.invoke(prompt)

    return AskResponse(
        answer=response.content,
        sources=chunks
    )