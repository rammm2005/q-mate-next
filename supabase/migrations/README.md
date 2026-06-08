# Supabase Migrations

## 001_initial_schema.sql

Creates the initial database schema for CodeQ-Mate including:

### Extensions
- **pgvector** - Enables vector similarity search for semantic retrieval
- **pg_trgm** - Enables trigram-based text search optimization

### Tables

#### `users`
Stores user accounts with API key authentication and access control.
- `api_key` - Unique API key for authentication
- `access_control_list` - Array of repository IDs the user can access
- `is_admin` - Admin flag for elevated permissions

#### `repositories`
Stores repository metadata and ingestion status.
- Tracks ingestion state, total chunks/files, languages detected
- Referenced by code_chunks via foreign key

#### `code_chunks`
Primary table storing all indexed code chunks with embeddings.
- All fields from the `CodeChunk` data model
- All metadata fields from `ChunkMetadata` (function_name, class_name, module_name, imports, dependencies, docstring, signatures, tags)
- `embedding` - 384-dimensional vector (all-MiniLM-L6-v2)
- `bm25_terms` - Tokenized terms for BM25 lexical search

### Indexes
- **HNSW** on `embedding` column for fast approximate nearest neighbor search
- **GIN** on `bm25_terms` for inverted index / lexical search
- **GIN** on `tags` for tag-based filtering
- Standard B-tree indexes on `repo_id`, `language`, `chunk_type`, `file_path`
- Composite index on `(repo_id, language)` for common filter patterns

### Functions
- `update_updated_at_column()` - Trigger function for automatic timestamp updates
- `match_chunks()` - Vector similarity search function with optional filters

### Security
- Row Level Security (RLS) enabled on `code_chunks` and `repositories` tables
