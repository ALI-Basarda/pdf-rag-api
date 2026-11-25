"""خدمة REST لاستهلاكها من OutSystems.

التشغيل:  uvicorn api:app --host 0.0.0.0 --port 8000
مواصفات OpenAPI:  http://<host>:8000/openapi.json  (تُستورد في OutSystems كـ Consume REST API)
"""
import io
import os
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from rag import PDFIndex, answer_with_claude, chunk_pages, extract_pages

app = FastAPI(title="PDF RAG API", version="1.0")
INDEXES: dict[str, PDFIndex] = {}  # في الذاكرة؛ للإنتاج استبدلها بـ Chroma/pgvector

API_KEY = os.getenv("RAG_API_KEY")  # اختياري: إن ضُبط يجب إرساله في ترويسة X-API-Key


def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")


class UploadResult(BaseModel):
    DocumentId: str
    Pages: int
    Chunks: int


class QueryRequest(BaseModel):
    DocumentId: str
    Question: str
    TopK: int = Field(5, ge=1, le=20)
    GenerateAnswer: bool = True


class Passage(BaseModel):
    Page: int
    Score: float
    Text: str


class QueryResult(BaseModel):
    Answer: str = ""  # فارغ إن لم يوجد مفتاح Claude أو GenerateAnswer=false
    Passages: list[Passage]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResult, dependencies=[Depends(check_key)])
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        pages = extract_pages(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid PDF file")
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "No extractable text (scanned PDF? run OCR first)")
    doc_id = uuid.uuid4().hex
    INDEXES[doc_id] = PDFIndex(chunks)
    return UploadResult(DocumentId=doc_id, Pages=len(pages), Chunks=len(chunks))


@app.post("/documents/raw", response_model=UploadResult, dependencies=[Depends(check_key)])
async def upload_raw(request: Request):
    """لـ OutSystems: الجسم هو ملف الـ PDF نفسه (Request Format = Binary)."""
    data = await request.body()
    try:
        pages = extract_pages(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid PDF file")
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "No extractable text (scanned PDF? run OCR first)")
    doc_id = uuid.uuid4().hex
    INDEXES[doc_id] = PDFIndex(chunks)
    return UploadResult(DocumentId=doc_id, Pages=len(pages), Chunks=len(chunks))


@app.post("/query", response_model=QueryResult, dependencies=[Depends(check_key)])
def query(req: QueryRequest):
    index = INDEXES.get(req.DocumentId)
    if not index:
        raise HTTPException(404, "Unknown DocumentId")
    hits = index.search(req.Question, req.TopK)
    answer = (answer_with_claude(req.Question, hits) if req.GenerateAnswer else None) or ""
    return QueryResult(
        Answer=answer,
        Passages=[Passage(Page=c.page, Score=s, Text=c.text) for c, s in hits],
    )
