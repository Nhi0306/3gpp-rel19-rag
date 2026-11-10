FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts
COPY data ./data

# BM25 stats la du lieu bat buoc de chay nhanh sparse; khong nam trong Rel-19/ khi clone
RUN mkdir -p Rel-19/rag && \
    python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('data/bm25-stats.json.gz', 'rb'), open('Rel-19/rag/bm25-stats.json', 'wb'))"

EXPOSE 7860

# Render/HF deu truyen PORT; mac dinh 7860 cho Hugging Face Spaces
CMD ["sh", "-c", "python scripts/rag_serve.py --host 0.0.0.0 --port ${PORT:-7860}"]
