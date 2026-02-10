"""
Main entry point for RAG API server
"""
# Enable truststore for Windows SSL certificate handling
import truststore
truststore.inject_into_ssl()

import uvicorn
import logging
from app.config import settings
from app.api import app  # app already has lifespan handler for initialization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )
