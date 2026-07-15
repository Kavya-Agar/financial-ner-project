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
# MODEL_DIR can point at a local models/... directory (baked into a custom
# image or mounted) or at a Hugging Face Hub model repo id (e.g.
# "kavyaagar/financial-ner-extended") - transformers' from_pretrained
# resolves either transparently, so no code change is needed between the
# two - only the env var differs.
ENV MODEL_DIR=models/financial-ner-extended
ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "backend.app:app"]
