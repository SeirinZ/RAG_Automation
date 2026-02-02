"""
Configuration settings for RAG system - OPTIMIZED FOR QWEN2.5:14B
"""
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # Directories
    CHROMA_DIR: str = "./chroma_db"
    CHROMA_PATH: Path = Path("./chroma_db")
    COLLECTION_NAME: str = "rag_documents"
    UPLOAD_DIR: str = "./pdf_uploads"
    PROCESSED_DIR: str = "./processed_pdfs"

    # ============================================================
    # MODEL SETTINGS - FULLY OPTIMIZED FOR QWEN2.5:14B
    # ============================================================

    # Embedding: E5-base multilingual - Better semantic understanding
    # Uses query/passage prefixes via E5EmbeddingWrapper in models.py
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"

    # LLM: llama3.1:8b via Ollama (faster, lighter)
    LLM_MODEL: str = "llama3.1:8b"
    LLM_TEMPERATURE: float = 0.0      # Zero for deterministic, factual answers
    LLM_NUM_CTX: int = 8192           # 8k context for 8b model

    # Reranker: Cross-encoder for precise relevance scoring
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ============================================================
    # RETRIEVAL SETTINGS - MAXIMIZED FOR 14B CAPABILITIES
    # ============================================================

    # Chunking: Smaller chunks with high overlap for RCA form documents
    CHUNK_SIZE: int = 1200            # Balanced for RCA content
    CHUNK_OVERLAP: int = 200          # Overlap to prevent splitting content

    # RCA-specific settings
    EXTRACT_TABLES: bool = False      # Disable table extraction for RCA forms (creates garbage)
    PREPEND_CONTEXT: bool = True      # Prepend AR# and title to each chunk

    # Retrieval: Balanced for speed and comprehensiveness
    TOP_K: int = 10                   # 10 chunks - good balance

    # Hybrid search weights
    SEMANTIC_WEIGHT: float = 0.6      # Slightly favor semantic (better embeddings now)
    KEYWORD_WEIGHT: float = 0.4       # Still important for exact term matching

    # Reranking: Wider net, stricter selection
    USE_RERANKING: bool = True
    RERANK_TOP_K: int = 80            # Cast wider net before reranking
    RELEVANCE_THRESHOLD: float = -6.0 # More permissive to catch relevant chunks
    MIN_CHUNKS: int = 7               # More chunks for completeness

    # ============================================================
    # API SETTINGS 
    # ============================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

# Create directories
Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(exist_ok=True)
Path(settings.CHROMA_DIR).mkdir(exist_ok=True)

# ============================================================
# PROMPT TEMPLATE - OPTIMIZED FOR RCA DOCUMENTS
# ============================================================
RAG_PROMPT_TEMPLATE = """You are a Root Cause Analysis (RCA) documentation expert. Answer the question using ONLY the provided context.

CONTEXT (from multiple sources):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Use ONLY information explicitly stated in the context above
2. Pay attention to source attribution - each source has [Source X: filename, page Y]
3. When asked about a specific AR/case, ONLY use information from that AR's document
4. For chronological events: Present in time order with dates/times
5. For root causes: List all contributing factors mentioned
6. For CAPA/actions: Include responsible persons and deadlines if mentioned
7. Preserve exact:
   - AR numbers (e.g., AR# 117422)
   - Equipment tags (e.g., PDS#1, KV-4106-1F)
   - Dates and times
   - Technical parameters
8. If information is not found, state: "The requested information was not found in the provided context."
9. If multiple documents discuss the same topic, synthesize but note which source each fact comes from

ANSWER:"""
