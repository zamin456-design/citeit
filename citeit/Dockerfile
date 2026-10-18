FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    BACKEND_URL=http://localhost:8000 \
    HF_HOME=/tmp/hf_cache

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the first request is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p backend/data/uploads backend/data/indexes && chmod -R 777 backend/data /tmp

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["./start.sh"]
