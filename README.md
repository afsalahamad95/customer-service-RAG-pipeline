# RAG Customer Service Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline for customer service automation with human-in-the-loop capabilities.

## Features

- **Preprocessing Pipeline**: Text cleaning, PII detection/removal, language detection, tokenization
- **Hybrid Retrieval**: Combines BM25 keyword search and semantic vector search
- **Intelligent Routing**: Confidence scoring, sentiment analysis, sensitive topic detection
- **LLM Integration**: Pluggable LLM support (local llama.cpp with Metal, OpenAI API)
- **Vector Storage**: PostgreSQL with pgvector extension
- **Metrics & Monitoring**: Built-in metrics tracking and health endpoints
- **RESTful API**: FastAPI-based API for integration

## Architecture

```
Input Query → Preprocessing → Retrieval → Decision → Response
                                            ↓
                                      Human Agent (if needed)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- (Optional) LLM model file for local inference

### Installation

1. Clone the repository and navigate to the project directory

2. Copy environment template:
```bash
cp .env.example .env
```

3. Edit `.env` and configure:
   - Database credentials
   - LLM provider (local or API)
   - API keys (if using OpenAI/Anthropic)

4. Start services with Docker:
```bash
docker-compose up -d
```

This will start:
- PostgreSQL with pgvector (port 5432)
- RAG Pipeline API (port 8000)

### Initialize Knowledge Base

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Initialize KB with sample data
python scripts/init_kb.py
```

## Using the API

### Submit a Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "top_k": 5
  }'
```

Response:
```json
{
  "query_id": 1,
  "response": "To reset your password, go to the login page...",
  "routing_decision": "auto_response",
  "confidence": 0.87,
  "sentiment": {"label": "neutral", "score": 0.95},
  "latency_ms": 342
}
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/kb/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "title": "New FAQ",
        "content": "Question and answer content...",
        "metadata": {"category": "faq"}
      }
    ]
  }'
```

### Get Metrics

```bash
curl http://localhost:8000/metrics
```

## Configuration

Edit `config.yaml` to customize:

- **Preprocessing**: Enable/disable PII removal, supported languages
- **Retrieval**: Embedding models, hybrid search weights, top-k
- **Decision**: Confidence thresholds, routing rules
- **LLM**: Provider, model parameters, prompts
- **Metrics**: Tracking intervals, export settings

## Local LLM Setup (M4 Pro / Apple Silicon)

1. Download a quantized Llama 3 model:
```bash
mkdir -p models
cd models
# Download from Hugging Face (example):
wget https://huggingface.co/.../llama-3-8b-instruct.Q4_K_M.gguf
```

2. Update `.env`:
```env
LLM_PROVIDER=local
LOCAL_MODEL_PATH=/models/llama-3-8b-instruct.Q4_K_M.gguf
LOCAL_MODEL_N_GPU_LAYERS=1  # Enable Metal acceleration
```

3. Install llama-cpp-python with Metal support:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Development

### Project Structure

```
RAG-project/
├── src/
│   ├── api/            # FastAPI application
│   ├── preprocessing/  # Text cleaning, PII removal, etc.
│   ├── retrieval/      # BM25, embeddings, vector store
│   ├── decision/       # Routing, confidence, sentiment
│   ├── response/       # LLM integration
│   ├── orchestration/  # Pipeline executor
│   ├── database/       # DB connection
│   └── utils/          # Logging, config, exceptions
├── database/           # SQL schema and migrations
├── data/               # Sample data
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── config.yaml         # Configuration
├── docker-compose.yml  # Docker setup
└── requirements.txt    # Python dependencies
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Monitoring

The pipeline tracks key metrics:
- Retrieval precision@k
- Deflection ratio (auto vs human)
- Latency (p95, p99)
- Sentiment classification accuracy
- Agent acceptance ratio

Access metrics at: http://localhost:8000/metrics

## Production Deployment

For production:

1. Use a proper secrets manager for API keys
2. Configure proper database backups
3. Set up monitoring (Prometheus + Grafana)
4. Use a reverse proxy (nginx)
5. Enable HTTPS
6. Scale workers based on load
7. Consider using a managed vector DB for large scale

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.
