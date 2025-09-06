from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..services.monitoring_services import (
    PerformanceTrackingService,
    AlertingService,
    RebalancingService,
    MarketRegimeDetectionService
)
from ..core.config import settings
from .base_agent import BaseAgent

class MonitoringAgent(BaseAgent):
    """Agent responsible for monitoring performance and adjusting strategies"""
    
    def __init__(self):
        super().__init__("MonitoringAgent")
        self.monitoring_services = {}
        self.llm = None
        self.agent_executor = None
        self.active_monitors = {}
        self.is_monitoring = False
    
    async def initialize(self):
        """Initialize the monitoring agent and monitoring services"""
        try:
            logger.info("Initializing Monitoring Agent")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1  # Low temperature for consistent monitoring
            )
            
            # Initialize monitoring services
            self.monitoring_services = {
                "performance_tracking": PerformanceTrackingService(),
                "alerting": AlertingService(),
                "rebalancing": RebalancingService(),
                "market_regime": MarketRegimeDetectionService()
            }
            
            # Initialize each service
            for name, service in self.monitoring_services.items():
                await service.initialize()
                logger.info(f"{name} monitoring service initialized")
            
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", "{input}"),
                ("assistant", "I'll continuously monitor the investment strategy performance and make necessary adjustments based on market conditions and performance metrics.")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            self.is_initialized = True
            logger.info("Monitoring Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Monitoring Agent: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the monitoring agent"""
        return """
        You are an expert monitoring agent responsible for continuous performance tracking and strategy adjustment.
        
        Your capabilities include:
        - Real-time performance monitoring and analysis
        - Market regime detection and adaptation
        - Automated alerting for significant events
        - Portfolio rebalancing recommendations
        - Risk monitoring and adjustment
        - Strategy performance evaluation
        - Adaptive learning from market feedback
        
        When monitoring investment strategies, you should:
        1. Track key performance metrics continuously
        2. Detect changes in market conditions and regimes
        3. Identify when strategies are underperforming or at risk
        4. Generate timely alerts for significant events
        5. Recommend adjustments based on performance data
        6. Monitor risk levels and suggest mitigation measures
        7. Learn from outcomes to improve future strategies
        8. Maintain detailed logs of all monitoring activities
        
        Always prioritize risk management and provide clear, actionable recommendations.
        Be proactive in identifying potential issues before they become significant problems.
        """
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the monitoring agent"""
        tools = [
            Tool(
                name="track_performance",
                description="Track and analyze investment performance metrics",
                func=self._track_performance
            ),
            Tool(
                name="detect_market_regime",
                description="Detect changes in market conditions and regimes",
                func=self._detect_market_regime
            ),
            Tool(
                name="generate_alerts",
                description="Generate alerts for significant events or threshold breaches",
                func=self._generate_alerts
            ),
            Tool(
                name="assess_rebalancing_needs",
                description="Assess if portfolio rebalancing is needed",
                func=self._assess_rebalancing_needs
            ),
            Tool(
                name="monitor_risk_levels",
                description="Monitor current risk levels and suggest adjustments",
                func=self._monitor_risk_levels
            ),
            Tool(
                name="evaluate_strategy_performance",
                description="Evaluate overall strategy performance and effectiveness",
                func=self._evaluate_strategy_performance
            )
        ]
        return tools
    
    async def _track_performance(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track and analyze performance metrics"""
        try:
            result = await self.monitoring_services["performance_tracking"].track_performance(portfolio_data)
            return {
                "monitoring_type": "performance_tracking",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in performance tracking: {e}")
            return {"error": str(e), "monitoring_type": "performance_tracking"}
    
    async def _detect_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in market conditions"""
        try:
            result = await self.monitoring_services["market_regime"].detect_regime_change(market_data)
            return {
                "monitoring_type": "market_regime",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return {"error": str(e), "monitoring_type": "market_regime"}
    
    async def _generate_alerts(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts for significant events"""
        try:
            result = await self.monitoring_services["alerting"].generate_alerts(monitoring_data)
            return {
                "monitoring_type": "alerting",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in alert generation: {e}")
            return {"error": str(e), "monitoring_type": "alerting"}
    
    async def _assess_rebalancing_needs(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if portfolio rebalancing is needed"""
        try:
            result = await self.monitoring_services["rebalancing"].assess_rebalancing_needs(portfolio_data)
            return {
                "monitoring_type": "rebalancing_assessment",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in rebalancing assessment: {e}")
            return {"error": str(e), "monitoring_type": "rebalancing_assessment"}
    
    async def _monitor_risk_levels(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor current risk levels"""
        try:
            # Calculate current risk metrics
            risk_metrics = await self.monitoring_services["performance_tracking"].calculate_risk_metrics(portfolio_data)
            
            # Check against risk thresholds
            risk_alerts = await self.monitoring_services["alerting"].check_risk_thresholds(risk_metrics)
            
            result = {
                "current_risk_metrics": risk_metrics,
                "risk_alerts": risk_alerts,
                "risk_status": self._assess_risk_status(risk_metrics)
            }
            
            return {
                "monitoring_type": "risk_monitoring",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}")
            return {"error": str(e), "monitoring_type": "risk_monitoring"}
    
    async def _evaluate_strategy_performance(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall strategy performance"""
        try:
            # Get performance metrics
            performance_metrics = await self.monitoring_services["performance_tracking"].get_strategy_performance(strategy_data)
            
            # Compare against benchmarks
            benchmark_comparison = await self.monitoring_services["performance_tracking"].compare_to_benchmark(performance_metrics)
            
            # Assess strategy effectiveness
            effectiveness_score = self._calculate_effectiveness_score(performance_metrics, benchmark_comparison)
            
            result = {
                "performance_metrics": performance_metrics,
                "benchmark_comparison": benchmark_comparison,
                "effectiveness_score": effectiveness_score,
                "recommendations": self._generate_performance_recommendations(performance_metrics, effectiveness_score)
            }
            
            return {
                "monitoring_type": "strategy_evaluation",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in strategy evaluation: {e}")
            return {"error": str(e), "monitoring_type": "strategy_evaluation"}
    
    def _assess_risk_status(self, risk_metrics: Dict[str, Any]) -> str:
        """Assess overall risk status based on metrics"""
        try:
            # Define risk thresholds
            high_risk_indicators = 0
            
            if risk_metrics.get("volatility", 0) > 0.25:  # 25% volatility
                high_risk_indicators += 1
            
            if risk_metrics.get("max_drawdown", 0) > 0.15:  # 15% max drawdown
                high_risk_indicators += 1
            
            if risk_metrics.get("var_95", 0) > 0.05:  # 5% VaR
                high_risk_indicators += 1
            
            if risk_metrics.get("beta", 1) > 1.5:  # High beta
                high_risk_indicators += 1
            
            if high_risk_indicators >= 3:
                return "HIGH_RISK"
            elif high_risk_indicators >= 2:
                return "MEDIUM_RISK"
            else:
                return "LOW_RISK"
                
        except Exception as e:
            logger.error(f"Error assessing risk status: {e}")
            return "UNKNOWN"
    
    def _calculate_effectiveness_score(self, performance_metrics: Dict[str, Any], benchmark_comparison: Dict[str, Any]) -> float:
        """Calculate strategy effectiveness score"""
        try:
            score = 0.0
            
            # Return performance (30%)
            excess_return = benchmark_comparison.get("excess_return", 0)
            if excess_return > 0.02:  # 2% outperformance
                score += 0.3
            elif excess_return > 0:
                score += 0.15
            
            # Risk-adjusted return (25%)
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.5:
                score += 0.25
            elif sharpe_ratio > 1.0:
                score += 0.15
            elif sharpe_ratio > 0.5:
                score += 0.1
            
            # Consistency (20%)
            max_drawdown = performance_metrics.get("max_drawdown", 1)
            if max_drawdown < 0.05:  # Less than 5% drawdown
                score += 0.2
            elif max_drawdown < 0.1:
                score += 0.15
            elif max_drawdown < 0.15:
                score += 0.1
            
            # Volatility management (15%)
            volatility = performance_metrics.get("volatility", 1)
            if volatility < 0.15:  # Less than 15% volatility
                score += 0.15
            elif volatility < 0.2:
                score += 0.1
            elif volatility < 0.25:
                score += 0.05
            
            # Win rate (10%)
            win_rate = performance_metrics.get("win_rate", 0.5)
            if win_rate > 0.6:
                score += 0.1
            elif win_rate > 0.55:
                score += 0.05
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating effectiveness score: {e}")
            return 0.0
    
    def _generate_performance_recommendations(self, performance_metrics: Dict[str, Any], effectiveness_score: float) -> List[str]:
        """Generate recommendations based on performance analysis"""
        try:
            recommendations = []
            
            if effectiveness_score < 0.3:
                recommendations.append("Consider major strategy revision - performance significantly below expectations")
            elif effectiveness_score < 0.5:
                recommendations.append("Review and adjust strategy parameters - moderate underperformance detected")
            elif effectiveness_score < 0.7:
                recommendations.append("Fine-tune strategy - minor adjustments may improve performance")
            else:
                recommendations.append("Strategy performing well - maintain current approach with regular monitoring")
            
            # Specific metric-based recommendations
            if performance_metrics.get("sharpe_ratio", 0) < 0.5:
                recommendations.append("Improve risk-adjusted returns by reducing volatility or increasing returns")
            
            if performance_metrics.get("max_drawdown", 0) > 0.15:
                recommendations.append("Implement stronger risk management to reduce maximum drawdown")
            
            if performance_metrics.get("volatility", 0) > 0.25:
                recommendations.append("Consider reducing position sizes or diversifying to lower volatility")
            
            if performance_metrics.get("win_rate", 0.5) < 0.45:
                recommendations.append("Review entry/exit criteria to improve win rate")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    async def execute(self, workflow_id: str, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring task for a specific workflow"""
        if not self.is_initialized:
            raise RuntimeError("Monitoring Agent not initialized")
        
        try:
            self._log_execution_start(f"monitoring workflow {workflow_id}")
            
            # Start continuous monitoring for this workflow
            monitor_task = asyncio.create_task(self._start_continuous_monitoring(workflow_id, strategy_results))
            self.active_monitors[workflow_id] = monitor_task
            
            # Perform initial monitoring assessment
            initial_assessment = await self._perform_initial_monitoring(workflow_id, strategy_results)
            
            monitoring_results = {
                "agent": self.name,
                "workflow_id": workflow_id,
                "strategy_summary": self._summarize_strategy(strategy_results),
                "initial_assessment": initial_assessment,
                "monitoring_status": "active",
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            self._log_execution_end(f"monitoring workflow {workflow_id}", success=True)
            return monitoring_results
            
        except Exception as e:
            self._log_execution_end(f"monitoring workflow {workflow_id}", success=False)
            return self._handle_error(e, f"monitoring workflow {workflow_id}")
    
    async def _start_continuous_monitoring(self, workflow_id: str, strategy_results: Dict[str, Any]):
        """Start continuous monitoring for a workflow"""
        try:
            logger.info(f"Starting continuous monitoring for workflow {workflow_id}")
            
            while workflow_id in self.active_monitors:
                # Perform monitoring cycle
                await self._monitoring_cycle(workflow_id, strategy_results)
                
                # Wait for next monitoring interval
                await asyncio.sleep(300)  # 5 minutes between monitoring cycles
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Error in continuous monitoring for workflow {workflow_id}: {e}")
    
    async def _monitoring_cycle(self, workflow_id: str, strategy_results: Dict[str, Any]):
        """Perform a single monitoring cycle"""
        try:
            # Get current portfolio data (this would come from a real data source)
            portfolio_data = await self._get_current_portfolio_data(workflow_id)
            
            # Track performance
            performance_result = await self._track_performance(portfolio_data)
            
            # Monitor risk levels
            risk_result = await self._monitor_risk_levels(portfolio_data)
            
            # Detect market regime changes
            market_result = await self._detect_market_regime(portfolio_data)
            
            # Generate alerts if needed
            alert_result = await self._generate_alerts({
                "performance": performance_result,
                "risk": risk_result,
                "market": market_result
            })
            
            # Log monitoring results
            logger.info(f"Monitoring cycle completed for workflow {workflow_id}")
            
            # Store monitoring results (this would go to a database in a real implementation)
            await self._store_monitoring_results(workflow_id, {
                "performance": performance_result,
                "risk": risk_result,
                "market": market_result,
                "alerts": alert_result,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle for workflow {workflow_id}: {e}")
    
    async def _perform_initial_monitoring(self, workflow_id: str, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial monitoring assessment"""
        try:
            # Extract strategy information
            strategy_summary = strategy_results.get("comprehensive_strategy", {})
            
            # Set up monitoring parameters
            monitoring_params = {
                "workflow_id": workflow_id,
                "strategy_type": strategy_summary.get("strategy_summary", {}).get("recommendation_strength", "Unknown"),
                "risk_level": strategy_summary.get("strategy_summary", {}).get("risk_level", "Medium"),
                "confidence_level": strategy_summary.get("confidence_level", 0.5)
            }
            
            # Initial assessment
            assessment = {
                "monitoring_setup": "completed",
                "monitoring_parameters": monitoring_params,
                "initial_risk_assessment": self._assess_initial_risk(strategy_results),
                "monitoring_frequency": "5 minutes",
                "alert_thresholds": self._set_alert_thresholds(monitoring_params)
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in initial monitoring assessment: {e}")
            return {"error": str(e)}
    
    async def _get_current_portfolio_data(self, workflow_id: str) -> Dict[str, Any]:
        """Get current portfolio data (placeholder for real implementation)"""
        # This would connect to real portfolio data sources
        return {
            "workflow_id": workflow_id,
            "total_value": 100000,  # Placeholder
            "positions": [],  # Placeholder
            "cash": 10000,  # Placeholder
            "timestamp": datetime.now().isoformat()
        }
    
    async def _store_monitoring_results(self, workflow_id: str, results: Dict[str, Any]):
        """Store monitoring results (placeholder for real implementation)"""
        # This would store results in a database
        logger.info(f"Storing monitoring results for workflow {workflow_id}")
    
    def _assess_initial_risk(self, strategy_results: Dict[str, Any]) -> str:
        """Assess initial risk level based on strategy"""
        try:
            strategy_summary = strategy_results.get("comprehensive_strategy", {}).get("strategy_summary", {})
            risk_level = strategy_summary.get("risk_level", "Medium")
            confidence_level = strategy_results.get("comprehensive_strategy", {}).get("confidence_level", 0.5)
            
            if risk_level == "High" or confidence_level < 0.4:
                return "HIGH_RISK"
            elif risk_level == "Low" and confidence_level > 0.8:
                return "LOW_RISK"
            else:
                return "MEDIUM_RISK"
                
        except Exception as e:
            logger.error(f"Error assessing initial risk: {e}")
            return "UNKNOWN"
    
    def _set_alert_thresholds(self, monitoring_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set alert thresholds based on monitoring parameters"""
        try:
            risk_level = monitoring_params.get("risk_level", "Medium")
            
            if risk_level == "High":
                thresholds = {
                    "max_drawdown": 0.10,  # 10%
                    "volatility": 0.30,     # 30%
                    "var_95": 0.08         # 8%
                }
            elif risk_level == "Low":
                thresholds = {
                    "max_drawdown": 0.05,  # 5%
                    "volatility": 0.15,     # 15%
                    "var_95": 0.03         # 3%
                }
            else:  # Medium
                thresholds = {
                    "max_drawdown": 0.08,  # 8%
                    "volatility": 0.20,     # 20%
                    "var_95": 0.05         # 5%
                }
            
            return thresholds
            
        except Exception as e:
            logger.error(f"Error setting alert thresholds: {e}")
            return {}
    
    def _summarize_strategy(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of strategy results for monitoring context"""
        try:
            return {
                "strategy_agent": strategy_results.get("agent"),
                "strategy_timestamp": strategy_results.get("timestamp"),
                "strategy_status": strategy_results.get("status"),
                "recommendation_strength": strategy_results.get("comprehensive_strategy", {}).get("strategy_summary", {}).get("recommendation_strength"),
                "confidence_level": strategy_results.get("comprehensive_strategy", {}).get("confidence_level"),
                "risk_level": strategy_results.get("comprehensive_strategy", {}).get("strategy_summary", {}).get("risk_level")
            }
        except Exception as e:
            logger.error(f"Error summarizing strategy: {e}")
            return {"error": str(e)}
    
    async def stop_monitoring(self, workflow_id: str):
        """Stop monitoring for a specific workflow"""
        try:
            if workflow_id in self.active_monitors:
                self.active_monitors[workflow_id].cancel()
                del self.active_monitors[workflow_id]
                logger.info(f"Stopped monitoring for workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Error stopping monitoring for workflow {workflow_id}: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop all active monitoring tasks
            for workflow_id in list(self.active_monitors.keys()):
                await self.stop_monitoring(workflow_id)
            
            # Cleanup monitoring services
            for service in self.monitoring_services.values():
                await service.cleanup()
            
            logger.info("Monitoring Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during Monitoring Agent cleanup: {e}")