# Dockerfile (backend)
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY backend /app
ENV PORT=5000
EXPOSE 5000
CMD ["python","app.py"]
