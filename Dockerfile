FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install dependencies first for faster caching
COPY fake_news_classifier/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (ignoring data/ and virtualenvs via .dockerignore)
COPY fake_news_classifier/ ./fake_news_classifier/

WORKDIR /app/fake_news_classifier

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --worker-class gthread --timeout 120 app:app"]
