from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.models.database import Base
from app.core.database import get_database_url
from loguru import logger
import asyncio

async def create_tables():
    """Create all database tables"""
    try:
        # Create synchronous engine for table creation
        engine = create_engine(get_database_url())
        
        logger.info("Creating database tables...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        
        # Create indexes if they don't exist
        with engine.connect() as conn:
            # Additional indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_research_symbols ON research_data USING GIN (symbols);",
                "CREATE INDEX IF NOT EXISTS idx_analysis_symbols ON analysis_data USING GIN (symbols);",
                "CREATE INDEX IF NOT EXISTS idx_workflow_monitoring ON workflow_executions (monitoring_enabled) WHERE monitoring_enabled = true;",
                "CREATE INDEX IF NOT EXISTS idx_active_sessions ON user_sessions (is_active, last_activity) WHERE is_active = true;",
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except SQLAlchemyError as e:
                    logger.warning(f"Index creation warning: {e}")
        
        engine.dispose()
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise

async def drop_tables():
    """Drop all database tables (use with caution)"""
    try:
        engine = create_engine(get_database_url())
        
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        
        engine.dispose()
        logger.info("All tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise

async def reset_database():
    """Reset database by dropping and recreating all tables"""
    logger.warning("Resetting database - all data will be lost!")
    await drop_tables()
    await create_tables()
    logger.info("Database reset completed")

async def check_database_schema():
    """Check if database schema is up to date"""
    try:
        engine = create_engine(get_database_url())
        
        with engine.connect() as conn:
            # Check if main tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                AND table_name IN (
                    'research_data', 'analysis_data', 'validation_data', 
                    'strategy_data', 'monitoring_data', 'workflow_executions',
                    'system_metrics', 'agent_execution_logs', 'user_sessions'
                )
            """))
            
            existing_tables = [row[0] for row in result.fetchall()]
            expected_tables = [
                'research_data', 'analysis_data', 'validation_data',
                'strategy_data', 'monitoring_data', 'workflow_executions',
                'system_metrics', 'agent_execution_logs', 'user_sessions'
            ]
            
            missing_tables = set(expected_tables) - set(existing_tables)
            
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                return False
            else:
                logger.info("Database schema is up to date")
                return True
                
        engine.dispose()
        
    except Exception as e:
        logger.error(f"Failed to check database schema: {e}")
        return False

if __name__ == "__main__":
    # Run migrations
    asyncio.run(create_tables())