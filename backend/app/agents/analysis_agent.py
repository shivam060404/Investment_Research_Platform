from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..services.analysis_services import (
    TechnicalAnalysisService,
    FundamentalAnalysisService,
    SentimentAnalysisService,
    RiskAnalysisService
)
from ..core.config import settings
from .base_agent import BaseAgent

class AnalysisAgent(BaseAgent):
    """Agent responsible for performing quantitative and qualitative analysis"""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
        self.analysis_services = {}
        self.llm = None
        self.agent_executor = None
    
    async def initialize(self):
        """Initialize the analysis agent and analysis services"""
        try:
            logger.info("Initializing Analysis Agent")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.2
            )
            
            # Initialize analysis services
            self.analysis_services = {
                "technical": TechnicalAnalysisService(),
                "fundamental": FundamentalAnalysisService(),
                "sentiment": SentimentAnalysisService(),
                "risk": RiskAnalysisService()
            }
            
            # Initialize each service
            for name, service in self.analysis_services.items():
                await service.initialize()
                logger.info(f"{name} analysis service initialized")
            
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", "{input}"),
                ("assistant", "I'll perform comprehensive analysis on the research data using both quantitative and qualitative methods.")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            self.is_initialized = True
            logger.info("Analysis Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Analysis Agent: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the analysis agent"""
        return """
        You are an expert financial analysis agent responsible for performing comprehensive investment analysis.
        
        Your capabilities include:
        - Technical analysis using various indicators and chart patterns
        - Fundamental analysis of financial statements and ratios
        - Sentiment analysis of news, social media, and market data
        - Risk analysis and portfolio optimization
        - Quantitative modeling and statistical analysis
        - Qualitative assessment of business models and competitive positioning
        
        When analyzing research data, you should:
        1. Perform both technical and fundamental analysis
        2. Assess market sentiment and behavioral factors
        3. Evaluate risk factors and potential downside scenarios
        4. Identify key trends, patterns, and anomalies
        5. Provide quantitative metrics and qualitative insights
        6. Consider multiple timeframes and market conditions
        7. Highlight areas of uncertainty or conflicting signals
        
        Always provide clear reasoning for your analysis and quantify confidence levels where possible.
        Consider both bullish and bearish scenarios in your assessment.
        """
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the analysis agent"""
        tools = [
            Tool(
                name="technical_analysis",
                description="Perform technical analysis including indicators, patterns, and price action",
                func=self._perform_technical_analysis
            ),
            Tool(
                name="fundamental_analysis",
                description="Perform fundamental analysis of financial statements and business metrics",
                func=self._perform_fundamental_analysis
            ),
            Tool(
                name="sentiment_analysis",
                description="Analyze market sentiment from news, social media, and market data",
                func=self._perform_sentiment_analysis
            ),
            Tool(
                name="risk_analysis",
                description="Assess investment risks and calculate risk metrics",
                func=self._perform_risk_analysis
            ),
            Tool(
                name="comparative_analysis",
                description="Compare investment against peers and benchmarks",
                func=self._perform_comparative_analysis
            ),
            Tool(
                name="scenario_analysis",
                description="Perform scenario and stress testing analysis",
                func=self._perform_scenario_analysis
            )
        ]
        return tools
    
    async def _perform_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on price data"""
        try:
            result = await self.analysis_services["technical"].analyze(data)
            return {
                "analysis_type": "technical",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"error": str(e), "analysis_type": "technical"}
    
    async def _perform_fundamental_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis on financial data"""
        try:
            result = await self.analysis_services["fundamental"].analyze(data)
            return {
                "analysis_type": "fundamental",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {"error": str(e), "analysis_type": "fundamental"}
    
    async def _perform_sentiment_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on textual data"""
        try:
            result = await self.analysis_services["sentiment"].analyze(data)
            return {
                "analysis_type": "sentiment",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e), "analysis_type": "sentiment"}
    
    async def _perform_risk_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk analysis and calculate risk metrics"""
        try:
            result = await self.analysis_services["risk"].analyze(data)
            return {
                "analysis_type": "risk",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {"error": str(e), "analysis_type": "risk"}
    
    async def _perform_comparative_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis against peers and benchmarks"""
        try:
            # Extract comparison data
            target_symbol = data.get("symbol")
            peers = data.get("peers", [])
            benchmark = data.get("benchmark", "^GSPC")
            
            # Perform comparative analysis using multiple services
            technical_comparison = await self.analysis_services["technical"].compare(target_symbol, peers, benchmark)
            fundamental_comparison = await self.analysis_services["fundamental"].compare(target_symbol, peers, benchmark)
            
            result = {
                "target_symbol": target_symbol,
                "peers": peers,
                "benchmark": benchmark,
                "technical_comparison": technical_comparison,
                "fundamental_comparison": fundamental_comparison
            }
            
            return {
                "analysis_type": "comparative",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return {"error": str(e), "analysis_type": "comparative"}
    
    async def _perform_scenario_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform scenario and stress testing analysis"""
        try:
            scenarios = data.get("scenarios", [
                {"name": "bull_case", "market_change": 0.20, "sector_change": 0.25},
                {"name": "base_case", "market_change": 0.08, "sector_change": 0.10},
                {"name": "bear_case", "market_change": -0.15, "sector_change": -0.20}
            ])
            
            scenario_results = []
            for scenario in scenarios:
                scenario_result = await self.analysis_services["risk"].stress_test(data, scenario)
                scenario_results.append({
                    "scenario": scenario["name"],
                    "parameters": scenario,
                    "result": scenario_result
                })
            
            return {
                "analysis_type": "scenario",
                "result": {
                    "scenarios": scenario_results,
                    "summary": self._summarize_scenarios(scenario_results)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {"error": str(e), "analysis_type": "scenario"}
    
    def _summarize_scenarios(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize scenario analysis results"""
        try:
            returns = [result["result"].get("expected_return", 0) for result in scenario_results]
            risks = [result["result"].get("risk_score", 0) for result in scenario_results]
            
            return {
                "expected_return_range": {"min": min(returns), "max": max(returns), "avg": np.mean(returns)},
                "risk_range": {"min": min(risks), "max": max(risks), "avg": np.mean(risks)},
                "best_case": max(scenario_results, key=lambda x: x["result"].get("expected_return", 0)),
                "worst_case": min(scenario_results, key=lambda x: x["result"].get("expected_return", 0))
            }
        except Exception as e:
            logger.error(f"Error summarizing scenarios: {e}")
            return {"error": str(e)}
    
    async def execute(self, research_data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute analysis task on research data"""
        if not self.is_initialized:
            raise RuntimeError("Analysis Agent not initialized")
        
        try:
            self._log_execution_start("comprehensive analysis")
            
            # Prepare analysis input
            analysis_input = {
                "input": f"Perform comprehensive analysis on the following research data: {research_data}",
                "research_data": research_data,
                "parameters": parameters or {}
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(analysis_input)
            
            # Structure the response
            analysis_results = {
                "agent": self.name,
                "research_data_summary": self._summarize_research_data(research_data),
                "analysis_results": result,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            self._log_execution_end("comprehensive analysis", success=True)
            return analysis_results
            
        except Exception as e:
            self._log_execution_end("comprehensive analysis", success=False)
            return self._handle_error(e, "analysis execution")
    
    def _summarize_research_data(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the research data for analysis"""
        try:
            return {
                "data_sources": list(research_data.get("results", {}).keys()) if "results" in research_data else [],
                "symbols_analyzed": self._extract_symbols(research_data),
                "data_timestamp": research_data.get("timestamp"),
                "data_quality": self._assess_data_quality(research_data)
            }
        except Exception as e:
            logger.error(f"Error summarizing research data: {e}")
            return {"error": str(e)}
    
    def _extract_symbols(self, research_data: Dict[str, Any]) -> List[str]:
        """Extract financial symbols from research data"""
        symbols = set()
        try:
            # Extract symbols from various parts of the research data
            if "results" in research_data:
                for key, value in research_data["results"].items():
                    if isinstance(value, dict) and "symbol" in value:
                        symbols.add(value["symbol"])
            return list(symbols)
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []
    
    def _assess_data_quality(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of research data"""
        try:
            quality_score = 0
            total_checks = 0
            issues = []
            
            # Check data recency
            if "timestamp" in research_data:
                data_age = (datetime.now() - datetime.fromisoformat(research_data["timestamp"].replace("Z", "+00:00"))).total_seconds() / 3600
                if data_age < 1:  # Less than 1 hour old
                    quality_score += 1
                elif data_age > 24:  # More than 24 hours old
                    issues.append("Data is more than 24 hours old")
                total_checks += 1
            
            # Check for errors in data
            if "results" in research_data:
                error_count = sum(1 for v in research_data["results"].values() if isinstance(v, dict) and "error" in v)
                if error_count == 0:
                    quality_score += 1
                else:
                    issues.append(f"{error_count} data sources returned errors")
                total_checks += 1
            
            # Check data completeness
            expected_sources = ["stock_data", "fundamentals", "news_data"]
            available_sources = list(research_data.get("results", {}).keys())
            completeness = len(set(available_sources) & set(expected_sources)) / len(expected_sources)
            if completeness > 0.8:
                quality_score += 1
            elif completeness < 0.5:
                issues.append("Missing critical data sources")
            total_checks += 1
            
            final_score = (quality_score / total_checks) if total_checks > 0 else 0
            
            return {
                "quality_score": final_score,
                "issues": issues,
                "completeness": completeness,
                "data_age_hours": data_age if "timestamp" in research_data else None
            }
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            for service in self.analysis_services.values():
                await service.cleanup()
            logger.info("Analysis Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during Analysis Agent cleanup: {e}")