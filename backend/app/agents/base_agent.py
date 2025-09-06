from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

class BaseAgent(ABC):
    """Base class for all AI agents in the investment research platform"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.created_at = datetime.now()
        self.last_execution = None
        self.execution_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent with necessary resources"""
        pass
    
    @abstractmethod
    async def execute(self, task_data: Any, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the agent's main task"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources when shutting down"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent"""
        return {
            "name": self.name,
            "is_initialized": self.is_initialized,
            "created_at": self.created_at.isoformat(),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "execution_count": self.execution_count,
            "error_count": self.error_count
        }
    
    def _update_execution_stats(self, success: bool = True):
        """Update execution statistics"""
        self.last_execution = datetime.now()
        self.execution_count += 1
        if not success:
            self.error_count += 1
    
    def _log_execution_start(self, task_description: str):
        """Log the start of task execution"""
        logger.info(f"{self.name} starting execution: {task_description}")
    
    def _log_execution_end(self, task_description: str, success: bool = True):
        """Log the end of task execution"""
        status = "completed" if success else "failed"
        logger.info(f"{self.name} {status} execution: {task_description}")
        self._update_execution_stats(success)
    
    def _handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle and log errors consistently"""
        error_msg = f"{self.name} error{' in ' + context if context else ''}: {str(error)}"
        logger.error(error_msg)
        self._update_execution_stats(success=False)
        
        return {
            "agent": self.name,
            "error": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }