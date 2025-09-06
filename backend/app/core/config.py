from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Agentic AI Investment Research Platform"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database Settings
    DATABASE_URL: str = "postgresql://user:password@localhost/investment_research"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Financial Data API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    BLOOMBERG_API_KEY: Optional[str] = None
    QUANDL_API_KEY: Optional[str] = None
    
    # Vector Database Settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    
    # Agent Settings
    MAX_CONCURRENT_AGENTS: int = 5
    AGENT_TIMEOUT: int = 300  # seconds
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Data Pipeline Settings
    DATA_REFRESH_INTERVAL: int = 300  # seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5  # seconds
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()