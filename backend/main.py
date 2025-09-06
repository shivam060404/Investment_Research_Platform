from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.database import init_db
from app.services.data_pipeline import DataPipeline
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.rag_system import RAGSystem

data_pipeline = None
knowledge_graph = None
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_pipeline, knowledge_graph, rag_system
    
    try:
        await init_db()
        
        data_pipeline = DataPipeline()
        knowledge_graph = KnowledgeGraphService()
        rag_system = RAGSystem()
        
        await data_pipeline.start()
        
        print("✅ Investment Research Platform started successfully")
        
    except Exception as e:
        print(f"❌ Failed to start services: {e}")
        raise
    
    yield
    
    try:
        if data_pipeline:
            await data_pipeline.stop()
        print("✅ Investment Research Platform shut down successfully")
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")

app = FastAPI(
    title="Investment Research Platform API",
    description="AI-powered investment research and analysis platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Investment Research Platform API is running"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )