from itertools import islice
from fastembed import SparseTextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseIndexParams, SparseVector, SparseVectorParams, VectorParams, PointStruct
from app.core.config import settings
import uuid

from app.core.dependencies import get_embeddings, get_qdrant, split_with_context
from app.core.schemas import IngestResponse

sparse_model = SparseTextEmbedding(model_name="qdrant/bm25")

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def ensure_collection_exists(client: QdrantClient):
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if settings.COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        ),
        print(f"Коллекция {settings.COLLECTION_NAME} создана")
    else:
        print(f"Коллекция {settings.COLLECTION_NAME} уже существует")

def ingest_pdf(file_path: str) -> dict:
    # 1. Загружаем PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    total_text = " ".join([d.page_content for d in documents]).strip()
    if len(total_text) < 50:
        raise ValueError(
            "PDF не содержит читаемого текста. "
            "Возможно это сканированный документ без OCR"
        )
    
    print(f"Загружено страниц: {len(documents)}")

    # 2. Разбиваем на чанки
    chunks = split_with_context(documents)
    print(f"Чанков создано: {len(chunks)}")

    # 3. Создаём embeddings
    embeddings_model = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embeddings_model.embed_documents(texts)
    print(f"Embeddings созданы: {len(dense_vectors)}")

    sparse_vectors = list(sparse_model.embed(texts))

    # 4. Сохраняем в Qdrant
    client = get_qdrant()
    ensure_collection_exists(client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vectors[i],
                "sparse": SparseVector(
                    indices=sparse_vectors[i].indices.tolist(),
                    values=sparse_vectors[i].values.tolist()
                )
            },
            payload={
                "text": chunks[i].page_content,
                "source": file_path,
                "page": chunks[i].metadata.get("page", 0)
            }
        )
        for i in range(len(chunks))
    ]

    print(f"Начинаю загрузку {len(points)} точек в Qdrant...")

    for batch in batched(points, 30):
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=batch
        )

    print("Загрузка завершена успешно")

    return IngestResponse(
        status= "success",
        pages=len(documents),
        chunks=len(chunks),
        collection=settings.COLLECTION_NAME
    )