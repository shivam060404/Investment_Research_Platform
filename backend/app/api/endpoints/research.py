from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ...agents.orchestrator import AgentOrchestrator
from ...models.requests import ResearchRequest
from ...models.responses import ResearchResponse
from ...core.database import get_db, get_redis

router = APIRouter()

@router.post("/execute", response_model=ResearchResponse)
async def execute_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    redis_client=Depends(get_redis)
):
    """Execute research task using the Research Agent"""
    try:
        logger.info(f"Executing research request: {request.query}")
        
        # TODO: Get orchestrator from dependency injection
        # For now, create a mock response
        result = {
            "data_sources": request.data_sources or ["yahoo_finance", "sec_filings"],
            "symbols": request.symbols or [],
            "findings": f"Research completed for: {request.query}",
            "confidence_score": 0.85,
            "sources_count": len(request.data_sources or ["yahoo_finance", "sec_filings"])
        }
        
        return ResearchResponse(
            request_id=request.request_id,
            query=request.query,
            results=result,
            status="completed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{request_id}")
async def get_research_status(
    request_id: str,
    redis_client=Depends(get_redis)
):
    """Get status of a research request"""
    try:
        # Check Redis for request status
        status = redis_client.get(f"research:{request_id}")
        if not status:
            raise HTTPException(status_code=404, detail="Research request not found")
        
        return {"request_id": request_id, "status": status.decode()}
        
    except Exception as e:
        logger.error(f"Error getting research status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_research_history(
    limit: int = 10,
    offset: int = 0,
    db=Depends(get_db)
):
    """Get research history"""
    try:
        # TODO: Implement database query for research history
        return {
            "total": 0,
            "limit": limit,
            "offset": offset,
            "results": []
        }
        
    except Exception as e:
        logger.error(f"Error getting research history: {e}")
        raise HTTPException(status_code=500, detail=str(e))