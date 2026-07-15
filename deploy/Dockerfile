FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_BASE_URL=""
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY src/ ./src/
COPY conftest.py ./
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860.
# This deploy variant points MODEL_DIR at the real Hub model repo instead
# of a local directory - transformers' from_pretrained resolves either
# transparently (see the root Dockerfile for the local-path default used
# by CI's docker-build check).
ENV MODEL_DIR=kavyaagar/financial-ner-extended
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "backend.app:app"]
