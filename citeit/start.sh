#!/bin/sh
# Runs FastAPI (internal, port 8000) and Streamlit (public, port 7860) in one container.
set -e
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
cd frontend
exec streamlit run app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
