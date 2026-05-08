from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import uvicorn
from app.core.dependencies import get_qdrant
from app.core.state import state
from app.core.validators import validate_pdf_upload
from app.services.ingest import ingest_pdf
from app.services.retriever import ask_question
from app.core.schemas import IngestResponse, AskResponse
import tempfile
from app.core.config import settings
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загружаем embedding модель...")
    state.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    print("Подключаемся к Qdrant...")
    state.qdrant = QdrantClient(
        url=settings.QDRANT_URL, 
        api_key=settings.QDRANT_API_KEY
    )
    
    print("Все ресурсы успешно инициализированы")
    
    yield
    
    if hasattr(state, "qdrant"):
        state.qdrant.close()
    print("Ресурсы очищены")

app = FastAPI(
    title="RAG PDF Chatbot",
    description="Загружай PDF и задавай вопросы по содержимому",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.APP_DOMEN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

async def verify_token(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != settings.APP_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")
    return x_api_key

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", dependencies=[Depends(verify_token)], response_model=IngestResponse)
async def upload_pdf(file: UploadFile = File(...)):
    
    content = await validate_pdf_upload(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return result

@app.post("/ask", dependencies=[Depends(verify_token)], response_model=AskResponse)
def ask(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Вопрос слишком длинный (макс. 1000 символов)")
    
    client = get_qdrant()
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if settings.COLLECTION_NAME not in names:
        raise HTTPException(
            status_code=404,
            detail="Документы ещё не загружены. Сначала загрузите PDF через /upload"
        )
    
    # 4. Коллекция пустая
    collection_info = client.get_collection(settings.COLLECTION_NAME)
    if collection_info.points_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Коллекция пуста. Загрузите PDF через /upload"
        )
    
    try:
        return ask_question(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
    
@app.delete("/clear", dependencies=[Depends(verify_token)])
async def clear_data():
    try:

        qdrant_client = get_qdrant()
        qdrant_client.delete_collection(collection_name=settings.COLLECTION_NAME)
        
        return {"status": "success", "message": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':

    port = int(os.getenv("PORT", 8000))

    uvicorn.run("app.main:app", host=settings.APP_HOST, port=port)