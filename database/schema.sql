-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Knowledge Base table with vector embeddings
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(384),  -- For MiniLM-L6-V2 (384 dimensions)
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx 
    ON knowledge_base USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Full-text search index for BM25
CREATE INDEX IF NOT EXISTS knowledge_base_content_idx 
    ON knowledge_base USING gin(to_tsvector('english', content));

-- Metadata index
CREATE INDEX IF NOT EXISTS knowledge_base_metadata_idx 
    ON knowledge_base USING gin(metadata);

-- Query history table
CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    preprocessed_text TEXT,
    language VARCHAR(10) DEFAULT 'en',
    detected_sentiment VARCHAR(50),
    confidence_score FLOAT,
    routing_decision VARCHAR(50),  -- auto_response, human_handoff
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),
    user_id VARCHAR(100)
);

-- Retrieved documents for each query
CREATE TABLE IF NOT EXISTS query_retrievals (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    kb_id INTEGER REFERENCES knowledge_base(id) ON DELETE SET NULL,
    retrieval_method VARCHAR(50),  -- bm25, semantic, hybrid
    score FLOAT,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Responses table
CREATE TABLE IF NOT EXISTS responses (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    response_text TEXT NOT NULL,
    response_type VARCHAR(50),  -- auto, human, hybrid
    generated_by VARCHAR(100),  -- llm_model or agent_id
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table for continuous learning
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    response_id INTEGER REFERENCES responses(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50),  -- thumbs_up, thumbs_down, agent_approval, correction
    feedback_value INTEGER,  -- 1 for positive, -1 for negative
    corrected_response TEXT,
    agent_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Human agent interactions
CREATE TABLE IF NOT EXISTS agent_interactions (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    action VARCHAR(50),  -- assigned, responded, approved, rejected
    time_spent_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Metrics aggregation table
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dimensions JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries(created_at);
CREATE INDEX IF NOT EXISTS queries_routing_decision_idx ON queries(routing_decision);
CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON feedback(created_at);
CREATE INDEX IF NOT EXISTS feedback_type_idx ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS metrics_name_timestamp_idx ON metrics(metric_name, timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for knowledge_base table
CREATE TRIGGER update_knowledge_base_updated_at 
    BEFORE UPDATE ON knowledge_base 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
