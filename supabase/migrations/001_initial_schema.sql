-- CodeQ-Mate: Initial database schema
-- Enables pgvector extension and creates tables for code chunks, embeddings, repositories, and users

-- ============================================================================
-- Extensions
-- ============================================================================

-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for text search optimization
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- Users Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    access_control_list TEXT[] NOT NULL DEFAULT '{}',  -- list of repo_ids user can access
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on api_key for fast authentication lookup
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users (api_key);

-- ============================================================================
-- Repositories Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    description TEXT,
    default_branch TEXT NOT NULL DEFAULT 'main',
    languages TEXT[] NOT NULL DEFAULT '{}',
    last_ingested_at TIMESTAMPTZ,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    total_files INTEGER NOT NULL DEFAULT 0,
    ingestion_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (ingestion_status IN ('pending', 'in_progress', 'completed', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on name for repository lookup
CREATE INDEX IF NOT EXISTS idx_repositories_name ON repositories (name);

-- ============================================================================
-- Code Chunks Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('typescript', 'javascript', 'python', 'php', 'go')),
    chunk_type TEXT NOT NULL CHECK (chunk_type IN (
        'function', 'class', 'method', 'documentation', 'module', 'config', 'api_endpoint'
    )),
    content TEXT NOT NULL CHECK (content != ''),
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    parent_id UUID REFERENCES code_chunks(id) ON DELETE SET NULL,

    -- Metadata fields from ChunkMetadata
    function_name TEXT,
    class_name TEXT,
    module_name TEXT,
    imports TEXT[] NOT NULL DEFAULT '{}',
    dependencies TEXT[] NOT NULL DEFAULT '{}',
    docstring TEXT,
    signatures TEXT[] NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',

    -- Embedding vector (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384),

    -- BM25 tokenized terms for lexical search
    bm25_terms TEXT[] NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- Indexes for Vector Search
-- ============================================================================

-- HNSW index for fast approximate nearest neighbor search on embeddings
-- HNSW provides better recall than IVFFlat and doesn't require training
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding_hnsw
    ON code_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Indexes for BM25 / Lexical Search
-- ============================================================================

-- GIN index on bm25_terms for fast term lookup in inverted index queries
CREATE INDEX IF NOT EXISTS idx_code_chunks_bm25_terms
    ON code_chunks
    USING gin (bm25_terms);

-- GIN index on tags for tag-based filtering
CREATE INDEX IF NOT EXISTS idx_code_chunks_tags
    ON code_chunks
    USING gin (tags);

-- ============================================================================
-- Indexes for Filtering
-- ============================================================================

-- Index on repo_id for repository-scoped queries
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_id ON code_chunks (repo_id);

-- Index on language for language filtering
CREATE INDEX IF NOT EXISTS idx_code_chunks_language ON code_chunks (language);

-- Index on chunk_type for type-based filtering
CREATE INDEX IF NOT EXISTS idx_code_chunks_chunk_type ON code_chunks (chunk_type);

-- Index on file_path for file path pattern matching
CREATE INDEX IF NOT EXISTS idx_code_chunks_file_path ON code_chunks (file_path);

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_language
    ON code_chunks (repo_id, language);

-- Index on parent_id for hierarchical chunk queries
CREATE INDEX IF NOT EXISTS idx_code_chunks_parent_id ON code_chunks (parent_id);

-- ============================================================================
-- Row Level Security (RLS)
-- ============================================================================

-- Enable RLS on code_chunks
ALTER TABLE code_chunks ENABLE ROW LEVEL SECURITY;

-- Enable RLS on repositories
ALTER TABLE repositories ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- Functions
-- ============================================================================

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_repositories_updated_at
    BEFORE UPDATE ON repositories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_code_chunks_updated_at
    BEFORE UPDATE ON code_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Vector similarity search function
-- ============================================================================

-- Function to perform semantic search with cosine similarity
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_count INTEGER DEFAULT 20,
    filter_repo_id UUID DEFAULT NULL,
    filter_language TEXT DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    id UUID,
    repo_id UUID,
    file_path TEXT,
    language TEXT,
    chunk_type TEXT,
    content TEXT,
    start_line INTEGER,
    end_line INTEGER,
    function_name TEXT,
    class_name TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cc.id,
        cc.repo_id,
        cc.file_path,
        cc.language,
        cc.chunk_type,
        cc.content,
        cc.start_line,
        cc.end_line,
        cc.function_name,
        cc.class_name,
        1 - (cc.embedding <=> query_embedding) AS similarity
    FROM code_chunks cc
    WHERE
        cc.embedding IS NOT NULL
        AND (filter_repo_id IS NULL OR cc.repo_id = filter_repo_id)
        AND (filter_language IS NULL OR cc.language = filter_language)
        AND (1 - (cc.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY cc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
