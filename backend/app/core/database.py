from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase
import redis
from loguru import logger

from .config import settings

def get_database_url() -> str:
    """Get database URL for SQLAlchemy"""
    return f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

# SQLAlchemy setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Neo4j setup
class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Redis setup
redis_client = redis.from_url(settings.REDIS_URL)

# Database instances
neo4j_db = Neo4jConnection(
    settings.NEO4J_URI,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD
)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_neo4j():
    """Dependency to get Neo4j connection"""
    return neo4j_db

def get_redis():
    """Dependency to get Redis client"""
    return redis_client

async def init_db():
    """Initialize database connections and create tables"""
    try:
        # Test PostgreSQL connection
        with engine.connect() as conn:
            logger.info("PostgreSQL connection established")
        
        # Test Neo4j connection
        neo4j_db.query("RETURN 1")
        logger.info("Neo4j connection established")
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection established")
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_db():
    """Close database connections"""
    try:
        neo4j_db.close()
        redis_client.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")