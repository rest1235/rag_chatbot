from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 10 * 1024 * 1024

async def validate_pdf_upload(file: UploadFile) -> bytes:
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой. Максимум 10MB")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Файл не является валидным PDF")

    return content