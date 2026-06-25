FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY research_agent/ ./research_agent/
COPY data/ ./data/
COPY vector_store/ ./vector_store/

ENV PYTHONPATH=/app
ENV GOOGLE_CLOUD_PROJECT=multi-agent-research-system
ENV GOOGLE_CLOUD_LOCATION=us-central1
ENV GEMINI_MODEL=gemini-2.5-flash
ENV EMBEDDING_MODEL=text-embedding-005

EXPOSE 8080

# ADK API server for Cloud Run
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080", "--with_ui", "research_agent"]
