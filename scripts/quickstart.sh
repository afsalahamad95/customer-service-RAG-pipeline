#!/bin/bash

# Quick start script for RAG Pipeline

echo "🚀 RAG Pipeline Quick Start"
echo "=============================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and configure your settings!"
    echo ""
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres

# Wait for PostgreSQL
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r ../requirements.txt

# Download spaCy model
echo "📥 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Initialize knowledge base
echo "💾 Initializing knowledge base with sample data..."
python scripts/init_kb.py

# Run basic tests
echo "🧪 Running basic tests..."
python scripts/test_pipeline.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env to configure LLM provider and API keys"
echo "2. Start the API server:"
echo "   python src/api/main.py"
echo ""
echo "Or use Docker:"
echo "   docker-compose up"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Swagger docs: http://localhost:8000/docs"
