from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..services.strategy_services import (
    PortfolioOptimizationService,
    RiskManagementService,
    AssetAllocationService,
    TradingStrategyService
)
from ..core.config import settings
from .base_agent import BaseAgent

class StrategyAgent(BaseAgent):
    """Agent responsible for generating actionable investment strategies and recommendations"""
    
    def __init__(self):
        super().__init__("StrategyAgent")
        self.strategy_services = {}
        self.llm = None
        self.agent_executor = None
    
    async def initialize(self):
        """Initialize the strategy agent and strategy services"""
        try:
            logger.info("Initializing Strategy Agent")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.3  # Moderate creativity for strategy generation
            )
            
            # Initialize strategy services
            self.strategy_services = {
                "portfolio_optimization": PortfolioOptimizationService(),
                "risk_management": RiskManagementService(),
                "asset_allocation": AssetAllocationService(),
                "trading_strategy": TradingStrategyService()
            }
            
            # Initialize each service
            for name, service in self.strategy_services.items():
                await service.initialize()
                logger.info(f"{name} strategy service initialized")
            
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", "{input}"),
                ("assistant", "I'll generate comprehensive investment strategies based on the validated analysis, considering risk management and portfolio optimization.")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            self.is_initialized = True
            logger.info("Strategy Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategy Agent: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the strategy agent"""
        return """
        You are an expert investment strategy agent responsible for generating actionable investment recommendations.
        
        Your capabilities include:
        - Portfolio optimization and asset allocation
        - Risk management and hedging strategies
        - Trading strategy development
        - Position sizing and timing recommendations
        - Multi-timeframe strategy coordination
        - Performance monitoring and adjustment protocols
        
        When generating investment strategies, you should:
        1. Consider the validated analysis results and confidence levels
        2. Incorporate appropriate risk management measures
        3. Provide specific, actionable recommendations with clear rationale
        4. Consider multiple investment timeframes (short, medium, long-term)
        5. Include position sizing and entry/exit criteria
        6. Address potential risks and mitigation strategies
        7. Provide performance monitoring guidelines
        8. Consider portfolio context and diversification needs
        
        Always provide clear reasoning for your recommendations and include specific metrics for success measurement.
        Consider both bullish and bearish scenarios in your strategy formulation.
        """
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the strategy agent"""
        tools = [
            Tool(
                name="optimize_portfolio",
                description="Optimize portfolio allocation based on analysis results",
                func=self._optimize_portfolio
            ),
            Tool(
                name="generate_trading_strategy",
                description="Generate specific trading strategies and entry/exit points",
                func=self._generate_trading_strategy
            ),
            Tool(
                name="assess_risk_management",
                description="Develop risk management and hedging strategies",
                func=self._assess_risk_management
            ),
            Tool(
                name="calculate_position_sizing",
                description="Calculate optimal position sizes based on risk tolerance",
                func=self._calculate_position_sizing
            ),
            Tool(
                name="create_asset_allocation",
                description="Create strategic asset allocation recommendations",
                func=self._create_asset_allocation
            ),
            Tool(
                name="develop_monitoring_plan",
                description="Develop performance monitoring and adjustment plan",
                func=self._develop_monitoring_plan
            )
        ]
        return tools
    
    async def _optimize_portfolio(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation based on analysis"""
        try:
            result = await self.strategy_services["portfolio_optimization"].optimize(analysis_data)
            return {
                "strategy_type": "portfolio_optimization",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {"error": str(e), "strategy_type": "portfolio_optimization"}
    
    async def _generate_trading_strategy(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific trading strategies"""
        try:
            result = await self.strategy_services["trading_strategy"].generate_strategy(analysis_data)
            return {
                "strategy_type": "trading_strategy",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in trading strategy generation: {e}")
            return {"error": str(e), "strategy_type": "trading_strategy"}
    
    async def _assess_risk_management(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop risk management strategies"""
        try:
            result = await self.strategy_services["risk_management"].assess_risks(analysis_data)
            return {
                "strategy_type": "risk_management",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in risk management assessment: {e}")
            return {"error": str(e), "strategy_type": "risk_management"}
    
    async def _calculate_position_sizing(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position sizes"""
        try:
            # Extract risk parameters
            risk_tolerance = analysis_data.get("risk_tolerance", 0.02)  # 2% default
            portfolio_value = analysis_data.get("portfolio_value", 100000)  # $100k default
            confidence_level = analysis_data.get("confidence_level", 0.7)
            
            result = await self.strategy_services["risk_management"].calculate_position_size(
                risk_tolerance, portfolio_value, confidence_level
            )
            
            return {
                "strategy_type": "position_sizing",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in position sizing calculation: {e}")
            return {"error": str(e), "strategy_type": "position_sizing"}
    
    async def _create_asset_allocation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategic asset allocation"""
        try:
            result = await self.strategy_services["asset_allocation"].create_allocation(analysis_data)
            return {
                "strategy_type": "asset_allocation",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in asset allocation: {e}")
            return {"error": str(e), "strategy_type": "asset_allocation"}
    
    async def _develop_monitoring_plan(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop performance monitoring plan"""
        try:
            monitoring_plan = {
                "key_metrics": [
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "volatility",
                    "beta"
                ],
                "monitoring_frequency": "daily",
                "rebalancing_triggers": [
                    {"metric": "deviation_from_target", "threshold": 0.05},
                    {"metric": "risk_level_change", "threshold": 0.1},
                    {"metric": "market_regime_change", "threshold": "significant"}
                ],
                "alert_conditions": [
                    {"condition": "drawdown > 10%", "action": "review_positions"},
                    {"condition": "volatility > 2x average", "action": "reduce_exposure"},
                    {"condition": "correlation breakdown", "action": "reassess_strategy"}
                ],
                "review_schedule": {
                    "daily": "performance_metrics",
                    "weekly": "risk_assessment",
                    "monthly": "strategy_review",
                    "quarterly": "full_rebalancing"
                }
            }
            
            return {
                "strategy_type": "monitoring_plan",
                "result": monitoring_plan,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in monitoring plan development: {e}")
            return {"error": str(e), "strategy_type": "monitoring_plan"}
    
    async def execute(self, validation_results: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute strategy generation task based on validation results"""
        if not self.is_initialized:
            raise RuntimeError("Strategy Agent not initialized")
        
        try:
            self._log_execution_start("investment strategy generation")
            
            # Prepare strategy input
            strategy_input = {
                "input": f"Generate comprehensive investment strategies based on the following validated analysis: {validation_results}",
                "validation_results": validation_results,
                "parameters": parameters or {}
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(strategy_input)
            
            # Generate comprehensive strategy recommendations
            strategy_recommendations = await self._generate_comprehensive_strategy(validation_results, parameters)
            
            # Structure the response
            strategy_results = {
                "agent": self.name,
                "validation_summary": self._summarize_validation(validation_results),
                "strategy_results": result,
                "comprehensive_strategy": strategy_recommendations,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            self._log_execution_end("investment strategy generation", success=True)
            return strategy_results
            
        except Exception as e:
            self._log_execution_end("investment strategy generation", success=False)
            return self._handle_error(e, "strategy generation")
    
    async def _generate_comprehensive_strategy(self, validation_results: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive investment strategy using all strategy services"""
        try:
            # Extract key information from validation results
            analysis_data = self._extract_analysis_data(validation_results)
            confidence_level = self._extract_confidence_level(validation_results)
            
            # Generate different strategy components
            strategy_components = {}
            
            # Portfolio optimization
            portfolio_opt = await self._optimize_portfolio(analysis_data)
            strategy_components["portfolio_optimization"] = portfolio_opt
            
            # Trading strategy
            trading_strategy = await self._generate_trading_strategy(analysis_data)
            strategy_components["trading_strategy"] = trading_strategy
            
            # Risk management
            risk_mgmt = await self._assess_risk_management(analysis_data)
            strategy_components["risk_management"] = risk_mgmt
            
            # Position sizing
            position_sizing = await self._calculate_position_sizing({
                **analysis_data,
                "confidence_level": confidence_level,
                "risk_tolerance": parameters.get("risk_tolerance", 0.02) if parameters else 0.02,
                "portfolio_value": parameters.get("portfolio_value", 100000) if parameters else 100000
            })
            strategy_components["position_sizing"] = position_sizing
            
            # Asset allocation
            asset_allocation = await self._create_asset_allocation(analysis_data)
            strategy_components["asset_allocation"] = asset_allocation
            
            # Monitoring plan
            monitoring_plan = await self._develop_monitoring_plan(strategy_components)
            strategy_components["monitoring_plan"] = monitoring_plan
            
            # Create final strategy summary
            strategy_summary = self._create_strategy_summary(strategy_components, confidence_level)
            
            return {
                "strategy_components": strategy_components,
                "strategy_summary": strategy_summary,
                "confidence_level": confidence_level,
                "implementation_timeline": self._create_implementation_timeline(strategy_components)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive strategy generation: {e}")
            return {"error": str(e)}
    
    def _extract_analysis_data(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analysis data from validation results"""
        try:
            analysis_data = {}
            
            if "analysis_summary" in validation_results:
                analysis_summary = validation_results["analysis_summary"]
                analysis_data.update(analysis_summary)
            
            if "validation_results" in validation_results:
                validation_data = validation_results["validation_results"]
                analysis_data["validation_data"] = validation_data
            
            return analysis_data
        except Exception as e:
            logger.error(f"Error extracting analysis data: {e}")
            return {}
    
    def _extract_confidence_level(self, validation_results: Dict[str, Any]) -> float:
        """Extract overall confidence level from validation results"""
        try:
            if "comprehensive_validation" in validation_results:
                comp_validation = validation_results["comprehensive_validation"]
                return comp_validation.get("overall_confidence", 0.5)
            return 0.5  # Default neutral confidence
        except Exception as e:
            logger.error(f"Error extracting confidence level: {e}")
            return 0.5
    
    def _create_strategy_summary(self, strategy_components: Dict[str, Any], confidence_level: float) -> Dict[str, Any]:
        """Create a comprehensive strategy summary"""
        try:
            # Determine overall recommendation based on confidence and analysis
            if confidence_level > 0.8:
                recommendation_strength = "Strong"
            elif confidence_level > 0.6:
                recommendation_strength = "Moderate"
            else:
                recommendation_strength = "Weak"
            
            # Extract key recommendations from components
            key_recommendations = []
            risk_level = "Medium"
            expected_return = "Market Average"
            
            # Analyze portfolio optimization results
            if "portfolio_optimization" in strategy_components:
                port_opt = strategy_components["portfolio_optimization"]
                if "result" in port_opt and not "error" in port_opt:
                    key_recommendations.append("Optimize portfolio allocation based on analysis")
            
            # Analyze trading strategy results
            if "trading_strategy" in strategy_components:
                trading = strategy_components["trading_strategy"]
                if "result" in trading and not "error" in trading:
                    key_recommendations.append("Implement systematic trading approach")
            
            # Analyze risk management
            if "risk_management" in strategy_components:
                risk_mgmt = strategy_components["risk_management"]
                if "result" in risk_mgmt and not "error" in risk_mgmt:
                    key_recommendations.append("Apply comprehensive risk management")
                    # Extract risk level if available
                    if isinstance(risk_mgmt["result"], dict):
                        risk_level = risk_mgmt["result"].get("risk_level", "Medium")
            
            return {
                "recommendation_strength": recommendation_strength,
                "confidence_level": confidence_level,
                "key_recommendations": key_recommendations,
                "risk_level": risk_level,
                "expected_return": expected_return,
                "investment_horizon": "Medium-term (6-18 months)",
                "success_metrics": [
                    "Outperform benchmark by 2-5%",
                    "Maintain Sharpe ratio > 1.0",
                    "Limit maximum drawdown to < 15%"
                ]
            }
        except Exception as e:
            logger.error(f"Error creating strategy summary: {e}")
            return {"error": str(e)}
    
    def _create_implementation_timeline(self, strategy_components: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation timeline for the strategy"""
        try:
            timeline = {
                "immediate": [
                    "Review current portfolio positions",
                    "Assess risk tolerance and constraints",
                    "Set up monitoring systems"
                ],
                "week_1": [
                    "Implement initial position adjustments",
                    "Establish risk management protocols",
                    "Begin systematic data collection"
                ],
                "month_1": [
                    "Complete portfolio rebalancing",
                    "Implement full trading strategy",
                    "Establish performance benchmarks"
                ],
                "ongoing": [
                    "Daily performance monitoring",
                    "Weekly risk assessment",
                    "Monthly strategy review",
                    "Quarterly rebalancing"
                ]
            }
            
            return timeline
        except Exception as e:
            logger.error(f"Error creating implementation timeline: {e}")
            return {"error": str(e)}
    
    def _summarize_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of validation results for strategy context"""
        try:
            return {
                "validation_agent": validation_results.get("agent"),
                "validation_timestamp": validation_results.get("timestamp"),
                "validation_status": validation_results.get("status"),
                "overall_confidence": validation_results.get("comprehensive_validation", {}).get("overall_confidence", 0.5),
                "key_validation_points": validation_results.get("comprehensive_validation", {}).get("validation_details", [])
            }
        except Exception as e:
            logger.error(f"Error summarizing validation: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            for service in self.strategy_services.values():
                await service.cleanup()
            logger.info("Strategy Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during Strategy Agent cleanup: {e}")