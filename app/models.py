"""
Machine learning models initialization - OPTIMIZED FOR QWEN2.5:14B + E5 EMBEDDINGS
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List
import logging

from .config import settings, RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class E5EmbeddingWrapper:
    """
    Custom wrapper for E5 embeddings that handles query/passage prefixing.
    E5 models require:
    - Queries prefixed with "query: "
    - Documents prefixed with "passage: "
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.is_e5_model = "e5" in model_name.lower()
        logger.info(f"  E5 model detected: {self.is_e5_model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with 'passage: ' prefix for E5"""
        if self.is_e5_model:
            texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query with 'query: ' prefix for E5"""
        if self.is_e5_model:
            text = f"query: {text}"
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist()


class ModelManager:
    """Manages all ML models - optimized for qwen2.5:14b"""

    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.reranker = None
        self.prompt = None

    def initialize(self):
        """Initialize all models"""

        # Use custom E5 wrapper for better embedding performance
        logger.info(f"🧮 Loading embedding model: {settings.EMBEDDING_MODEL}")

        if "e5" in settings.EMBEDDING_MODEL.lower():
            # Use custom E5 wrapper with proper prefixing
            self.embeddings = E5EmbeddingWrapper(settings.EMBEDDING_MODEL)
            logger.info("  ✓ Using E5 embedding wrapper with query/passage prefixes")
        else:
            # Standard HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        logger.info("✓ Embeddings loaded")

        logger.info(f"🤖 Connecting to Ollama: {settings.LLM_MODEL}")
        logger.info(f"  Context window: {settings.LLM_NUM_CTX} tokens")
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_ctx=settings.LLM_NUM_CTX
        )
        logger.info("✓ LLM connected")

        if settings.USE_RERANKING:
            logger.info(f"🎯 Loading reranker: {settings.RERANKER_MODEL}")
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
            logger.info("✓ Reranker loaded")

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        logger.info("✓ All models initialized")

    def rerank_documents(self, query: str, documents, top_k: int = None):
        """Rerank documents using cross-encoder"""

        if not self.reranker or not documents:
            return documents

        top_k = top_k or settings.TOP_K

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return just documents (not tuples) for compatibility
        return [doc for doc, score in doc_score_pairs[:top_k]]

# Global model manager
model_manager = ModelManager()
