from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from .endpoints import research, analysis, validation, strategy, monitoring, workflows
from ..core.database import get_db, get_neo4j, get_redis
from ..agents.orchestrator import AgentOrchestrator
from ..models.requests import (
    ResearchRequest,
    AnalysisRequest,
    ValidationRequest,
    StrategyRequest,
    MonitoringRequest,
    WorkflowRequest
)
from ..models.responses import (
    ResearchResponse,
    AnalysisResponse,
    ValidationResponse,
    StrategyResponse,
    MonitoringResponse,
    WorkflowResponse,
    StatusResponse
)

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(validation.router, prefix="/validation", tags=["validation"])
api_router.include_router(strategy.router, prefix="/strategy", tags=["strategy"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])

# Global orchestrator instance (this would be injected in a real application)
orchestrator = None

def get_orchestrator() -> AgentOrchestrator:
    """Dependency to get the agent orchestrator"""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Agent orchestrator not initialized")
    return orchestrator

def set_orchestrator(orch: AgentOrchestrator):
    """Set the global orchestrator instance"""
    global orchestrator
    orchestrator = orch

# Root endpoints
@api_router.get("/status", response_model=StatusResponse)
async def get_api_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get API and agent status"""
    try:
        agent_status = await orchestrator.get_agent_status()
        return StatusResponse(
            status="healthy",
            timestamp=datetime.now(),
            agents=agent_status,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/research/execute", response_model=ResearchResponse)
async def execute_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Execute research task using the Research Agent"""
    try:
        logger.info(f"Executing research request: {request.query}")
        
        result = await orchestrator.execute_single_agent(
            "research",
            {
                "query": request.query,
                "symbols": request.symbols,
                "data_sources": request.data_sources,
                "timeframe": request.timeframe
            }
        )
        
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

@api_router.post("/analysis/execute", response_model=AnalysisResponse)
async def execute_analysis(
    request: AnalysisRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Execute analysis task using the Analysis Agent"""
    try:
        logger.info(f"Executing analysis request for research data")
        
        result = await orchestrator.execute_single_agent(
            "analysis",
            {
                "research_data": request.research_data,
                "analysis_types": request.analysis_types,
                "parameters": request.parameters
            }
        )
        
        return AnalysisResponse(
            request_id=request.request_id,
            research_data_id=request.research_data_id,
            results=result,
            status="completed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/validation/execute", response_model=ValidationResponse)
async def execute_validation(
    request: ValidationRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Execute validation task using the Validation Agent"""
    try:
        logger.info(f"Executing validation request for analysis data")
        
        result = await orchestrator.execute_single_agent(
            "validation",
            {
                "analysis_data": request.analysis_data,
                "validation_types": request.validation_types,
                "parameters": request.parameters
            }
        )
        
        return ValidationResponse(
            request_id=request.request_id,
            analysis_data_id=request.analysis_data_id,
            results=result,
            status="completed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/strategy/execute", response_model=StrategyResponse)
async def execute_strategy(
    request: StrategyRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Execute strategy generation using the Strategy Agent"""
    try:
        logger.info(f"Executing strategy request for validation data")
        
        result = await orchestrator.execute_single_agent(
            "strategy",
            {
                "validation_data": request.validation_data,
                "strategy_types": request.strategy_types,
                "risk_tolerance": request.risk_tolerance,
                "investment_horizon": request.investment_horizon,
                "parameters": request.parameters
            }
        )
        
        return StrategyResponse(
            request_id=request.request_id,
            validation_data_id=request.validation_data_id,
            results=result,
            status="completed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/monitoring/start", response_model=MonitoringResponse)
async def start_monitoring(
    request: MonitoringRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Start monitoring for a strategy"""
    try:
        logger.info(f"Starting monitoring for workflow {request.workflow_id}")
        
        # Start monitoring in background
        background_tasks.add_task(
            orchestrator.execute_single_agent,
            "monitoring",
            {
                "workflow_id": request.workflow_id,
                "strategy_data": request.strategy_data,
                "monitoring_parameters": request.monitoring_parameters
            }
        )
        
        return MonitoringResponse(
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            status="monitoring_started",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/workflows/execute", response_model=WorkflowResponse)
async def execute_complete_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Execute complete investment research workflow"""
    try:
        logger.info(f"Executing complete workflow: {request.query}")
        
        # Execute the complete workflow
        result = await orchestrator.execute_research_workflow(
            request.query,
            {
                "symbols": request.symbols,
                "data_sources": request.data_sources,
                "analysis_types": request.analysis_types,
                "risk_tolerance": request.risk_tolerance,
                "investment_horizon": request.investment_horizon,
                "parameters": request.parameters
            }
        )
        
        return WorkflowResponse(
            request_id=request.request_id,
            workflow_id=result.get("workflow_id"),
            query=request.query,
            results=result,
            status=result.get("status", "completed"),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow_results(
    workflow_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get results for a specific workflow"""
    try:
        result = await orchestrator.get_workflow_results(workflow_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse(
            request_id=result.get("workflow_id"),
            workflow_id=workflow_id,
            query=result.get("query"),
            results=result,
            status=result.get("status"),
            timestamp=datetime.fromisoformat(result.get("timestamp"))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/workflows", response_model=List[str])
async def list_workflows(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """List all available workflows"""
    try:
        workflows = await orchestrator.list_workflows()
        return workflows
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/monitoring/{workflow_id}")
async def stop_monitoring(
    workflow_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Stop monitoring for a specific workflow"""
    try:
        # Get monitoring agent and stop monitoring
        monitoring_agent = orchestrator.agents.get("monitoring")
        if monitoring_agent:
            await monitoring_agent.stop_monitoring(workflow_id)
        
        return {"message": f"Monitoring stopped for workflow {workflow_id}"}
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoints
@api_router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Investment Research Platform API"
    }

@api_router.get("/health/detailed")
async def detailed_health_check(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Detailed health check including agent status"""
    try:
        agent_status = await orchestrator.get_agent_status()
        
        # Check if all agents are healthy
        all_healthy = all(status != "error" for status in agent_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service": "Investment Research Platform API",
            "agents": agent_status,
            "orchestrator_initialized": orchestrator.is_initialized
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Investment Research Platform API",
            "error": str(e)
        }