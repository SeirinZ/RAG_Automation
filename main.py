"""
Main entry point for RAG API server
"""
import uvicorn
import logging
from app.config import settings
from app.models import model_manager
from app.vector_store import vector_store
from app.retrieval import retriever
from app.api import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize all components"""
    
    logger.info("="*60)
    logger.info("🚀 STARTING RAG SYSTEM")
    logger.info("="*60)
    
    # Initialize models
    logger.info(">>> Step 1: Models")
    model_manager.initialize()
    
    # Initialize vector store - WAJIB ADA!
    logger.info(">>> Step 2: Vector Store")
    vector_store.initialize()
    
    # Initialize retrieval
    logger.info(">>> Step 3: Retrieval")
    retriever.initialize()
    
    logger.info("="*60)
    logger.info("✅ SYSTEM READY")
    logger.info("="*60)
    logger.info(f"API: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info("="*60)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_system()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )
