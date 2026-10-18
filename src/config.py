import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env from the project root
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class Config:
    """Enterprise Financial RAG Configuration.
    Supports seamless switching between Cloud LLMs (Anthropic, OpenAI, Cohere)
    and Zero-Cost Local Mode for immediate local development.
    """
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

    # Retrieval Hyperparameters
    DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
    SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "10"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
    RRF_K = int(os.getenv("RRF_K", "60"))  # Reciprocal Rank Fusion constant
    
    # Chunking Hyperparameters
    MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "400"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
    
    # Model Names
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "rerank-v3.5")

    @classmethod
    def is_anthropic_available(cls) -> bool:
        return bool(cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_API_KEY.startswith("sk-ant"))

    @classmethod
    def is_openai_available(cls) -> bool:
        return bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY.startswith("sk-"))

    @classmethod
    def is_cohere_available(cls) -> bool:
        return bool(cls.COHERE_API_KEY and len(cls.COHERE_API_KEY) > 10)
