from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from loguru import logger
from enum import Enum

from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .validation_agent import ValidationAgent
from .strategy_agent import StrategyAgent
from .monitoring_agent import MonitoringAgent
from ..core.config import settings

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

class AgentOrchestrator:
    """Orchestrates multiple AI agents for investment research"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results_cache: Dict[str, Any] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all agents"""
        try:
            logger.info("Initializing Agent Orchestrator")
            
            # Initialize agents
            self.agents = {
                "research": ResearchAgent(),
                "analysis": AnalysisAgent(),
                "validation": ValidationAgent(),
                "strategy": StrategyAgent(),
                "monitoring": MonitoringAgent()
            }
            
            # Initialize each agent
            for name, agent in self.agents.items():
                try:
                    await agent.initialize()
                    self.agent_status[name] = AgentStatus.IDLE
                    logger.info(f"{name.capitalize()} Agent initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize {name} agent: {e}")
                    self.agent_status[name] = AgentStatus.ERROR
            
            self.is_initialized = True
            logger.info("Agent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Orchestrator: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all agents"""
        try:
            logger.info("Cleaning up Agent Orchestrator")
            
            for name, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    self.agent_status[name] = AgentStatus.STOPPED
                except Exception as e:
                    logger.error(f"Error cleaning up {name} agent: {e}")
            
            logger.info("Agent Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {name: status.value for name, status in self.agent_status.items()}
    
    async def execute_research_workflow(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute complete research workflow"""
        if not self.is_initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting research workflow {workflow_id} for query: {query}")
        
        try:
            # Step 1: Research Agent - Gather data
            logger.info("Step 1: Gathering research data")
            self.agent_status["research"] = AgentStatus.RUNNING
            research_results = await self.agents["research"].execute(query, parameters)
            self.agent_status["research"] = AgentStatus.IDLE
            
            # Step 2: Analysis Agent - Analyze data
            logger.info("Step 2: Analyzing research data")
            self.agent_status["analysis"] = AgentStatus.RUNNING
            analysis_results = await self.agents["analysis"].execute(research_results, parameters)
            self.agent_status["analysis"] = AgentStatus.IDLE
            
            # Step 3: Validation Agent - Validate findings
            logger.info("Step 3: Validating analysis results")
            self.agent_status["validation"] = AgentStatus.RUNNING
            validation_results = await self.agents["validation"].execute(analysis_results, parameters)
            self.agent_status["validation"] = AgentStatus.IDLE
            
            # Step 4: Strategy Agent - Generate recommendations
            logger.info("Step 4: Generating investment strategy")
            self.agent_status["strategy"] = AgentStatus.RUNNING
            strategy_results = await self.agents["strategy"].execute(validation_results, parameters)
            self.agent_status["strategy"] = AgentStatus.IDLE
            
            # Compile final results
            final_results = {
                "workflow_id": workflow_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "research_data": research_results,
                "analysis": analysis_results,
                "validation": validation_results,
                "strategy": strategy_results,
                "status": "completed"
            }
            
            # Cache results
            self.results_cache[workflow_id] = final_results
            
            # Start monitoring
            asyncio.create_task(self._start_monitoring(workflow_id, strategy_results))
            
            logger.info(f"Research workflow {workflow_id} completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in research workflow {workflow_id}: {e}")
            # Reset agent statuses
            for agent_name in self.agents.keys():
                if self.agent_status[agent_name] == AgentStatus.RUNNING:
                    self.agent_status[agent_name] = AgentStatus.ERROR
            
            return {
                "workflow_id": workflow_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    async def _start_monitoring(self, workflow_id: str, strategy_results: Dict[str, Any]):
        """Start monitoring for a completed workflow"""
        try:
            self.agent_status["monitoring"] = AgentStatus.RUNNING
            await self.agents["monitoring"].execute(workflow_id, strategy_results)
        except Exception as e:
            logger.error(f"Error in monitoring for workflow {workflow_id}: {e}")
        finally:
            self.agent_status["monitoring"] = AgentStatus.IDLE
    
    async def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get cached workflow results"""
        return self.results_cache.get(workflow_id)
    
    async def list_workflows(self) -> List[str]:
        """List all cached workflow IDs"""
        return list(self.results_cache.keys())
    
    async def execute_single_agent(self, agent_name: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent task"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        if self.agent_status[agent_name] == AgentStatus.RUNNING:
            raise RuntimeError(f"Agent {agent_name} is already running")
        
        try:
            self.agent_status[agent_name] = AgentStatus.RUNNING
            result = await self.agents[agent_name].execute(task_data)
            self.agent_status[agent_name] = AgentStatus.IDLE
            return result
        except Exception as e:
            self.agent_status[agent_name] = AgentStatus.ERROR
            logger.error(f"Error executing {agent_name} agent: {e}")
            raise