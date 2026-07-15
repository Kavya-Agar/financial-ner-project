FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY src/data/label_schema.py ./src/data/label_schema.py
COPY src/postprocess/regex_rules.py ./src/postprocess/regex_rules.py
COPY conftest.py ./

# Backend-only image for Render: no frontend bundling (the frontend is
# deployed separately, e.g. to Vercel) and no static-file serving needed.
# MODEL_DIR points at the real Hub model repo, same as deploy/Dockerfile
# (the Hugging Face Space variant).
ENV MODEL_DIR=kavyaagar/financial-ner-extended

# Render (like most container platforms) injects PORT at runtime rather
# than honoring a fixed EXPOSE; fall back to 8000 for `docker run` without
# PORT set (e.g. local testing).
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120 backend.app:app"]
