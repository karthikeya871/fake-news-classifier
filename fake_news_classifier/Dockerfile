FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install dependencies first for faster Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models (excluding data/ via .dockerignore)
COPY . .

# Hugging Face Spaces uses 7860; Render/Railway injects PORT
EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --worker-class gthread --timeout 120 app:app"]
