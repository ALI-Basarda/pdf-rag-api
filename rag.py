"""RAG بسيط لملفات PDF: استخراج -> تقطيع -> تضمين -> بحث -> (اختياري) إجابة بواسطة Claude."""
import os
import re
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader

# نموذج تضمين متعدد اللغات (يدعم العربية والإنجليزية) بصيغة ONNX الخفيفة
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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


_MODEL = None


def _get_model():
    """يحمّل نموذج ONNX الخفيف مرة واحدة فقط (بدون torch لتوفير الذاكرة)."""
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding

        _MODEL = TextEmbedding(EMBED_MODEL)
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_get_model().embed(texts, batch_size=16)), dtype=np.float32)
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
