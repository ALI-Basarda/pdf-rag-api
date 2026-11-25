"""RAG بسيط لملفات PDF: استخراج -> تقطيع -> تضمين -> بحث -> (اختياري) إجابة بواسطة Claude."""
import os
import re
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader

# تضمين عبر OpenAI API (يدعم العربية والإنجليزية) بدون نموذج محلي لتبقى الذاكرة صغيرة
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


@dataclass
class Chunk:
    text: str
    page: int


def extract_pages(pdf_file) -> list[tuple[int, str]]:
    """يرجع [(رقم_الصفحة, النص)] من ملف PDF (مسار أو كائن ملف)."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """يقطّع كل صفحة إلى مقاطع متداخلة مع الحفاظ على رقم الصفحة."""
    chunks = []
    step = max(size - overlap, 1)
    for page_no, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + size].strip()
            if len(piece) > 30:
                chunks.append(Chunk(piece, page_no))
            if start + size >= len(text):
                break
    return chunks


def _embed(texts: list[str]) -> np.ndarray:
    """يرسل النصوص إلى OpenAI على دفعات ويرجع متجهات مطبّعة."""
    import httpx
    from fastapi import HTTPException

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(500, "OPENAI_API_KEY is not set on the server")
    out = []
    for i in range(0, len(texts), 96):
        r = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": EMBED_MODEL, "input": texts[i:i + 96]},
            timeout=60,
        )
        if r.status_code != 200:
            raise HTTPException(502, f"Embedding API error {r.status_code}: {r.text[:200]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    vecs = np.array(out, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class PDFIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = _embed([c.text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = _embed([query])[0]
        scores = self.vectors @ q  # الضرب النقطي = cosine لأن المتجهات مطبّعة
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def answer_with_claude(question: str, hits: list[tuple[Chunk, float]]) -> str | None:
    """يولّد إجابة اعتماداً على المقاطع المسترجعة فقط. يرجع None إن لم يوجد مفتاح API."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    context = "\n\n".join(f"[صفحة {c.page}]\n{c.text}" for c, _ in hits)
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=(
            "أجب عن السؤال اعتماداً على المقاطع المرفقة فقط، وبنفس لغة السؤال. "
            "اذكر رقم الصفحة عند الاستشهاد. إذا لم تجد الإجابة في المقاطع فقل ذلك صراحةً."
        ),
        messages=[{
            "role": "user",
            "content": f"المقاطع:\n{context}\n\nالسؤال: {question}",
        }],
    )
    return msg.content[0].text
