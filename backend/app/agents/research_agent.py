from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
from loguru import logger
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..services.data_sources import (
    YFinanceService,
    AlphaVantageService,
    SECFilingsService,
    NewsService,
    EarningsCallService
)
from ..core.config import settings
from .base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    """Agent responsible for gathering investment research data from multiple sources"""
    
    def __init__(self):
        super().__init__("ResearchAgent")
        self.data_services = {}
        self.llm = None
        self.agent_executor = None
    
    async def initialize(self):
        """Initialize the research agent and data services"""
        try:
            logger.info("Initializing Research Agent")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1
            )
            
            # Initialize data services
            self.data_services = {
                "yfinance": YFinanceService(),
                "alpha_vantage": AlphaVantageService(settings.ALPHA_VANTAGE_API_KEY),
                "sec_filings": SECFilingsService(),
                "news": NewsService(),
                "earnings_calls": EarningsCallService()
            }
            
            # Initialize each service
            for name, service in self.data_services.items():
                await service.initialize()
                logger.info(f"{name} service initialized")
            
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", "{input}"),
                ("assistant", "I'll help you gather comprehensive investment research data. Let me search through multiple financial data sources to find relevant information.")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            self.is_initialized = True
            logger.info("Research Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Research Agent: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the research agent"""
        return """
        You are an expert financial research agent responsible for gathering comprehensive investment data.
        
        Your capabilities include:
        - Collecting real-time and historical market data
        - Retrieving SEC filings and regulatory documents
        - Gathering financial news and sentiment data
        - Accessing earnings call transcripts and analysis
        - Performing fundamental and technical analysis data collection
        
        When given a research query, you should:
        1. Identify the relevant financial instruments (stocks, bonds, commodities, etc.)
        2. Gather data from multiple sources to ensure comprehensive coverage
        3. Collect both quantitative data (prices, volumes, ratios) and qualitative data (news, filings)
        4. Organize the data in a structured format for further analysis
        5. Note any data limitations or potential biases in your sources
        
        Always prioritize accuracy and recency of data. If data is unavailable or unreliable, clearly indicate this in your response.
        """
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the research agent"""
        tools = [
            Tool(
                name="get_stock_data",
                description="Get comprehensive stock data including price, volume, and financial metrics",
                func=self._get_stock_data
            ),
            Tool(
                name="get_company_fundamentals",
                description="Get company fundamental data including financial statements and ratios",
                func=self._get_company_fundamentals
            ),
            Tool(
                name="get_sec_filings",
                description="Get SEC filings for a company including 10-K, 10-Q, and 8-K reports",
                func=self._get_sec_filings
            ),
            Tool(
                name="get_financial_news",
                description="Get recent financial news and sentiment analysis for a company or sector",
                func=self._get_financial_news
            ),
            Tool(
                name="get_earnings_data",
                description="Get earnings call transcripts and earnings data",
                func=self._get_earnings_data
            ),
            Tool(
                name="get_market_data",
                description="Get broader market data and economic indicators",
                func=self._get_market_data
            )
        ]
        return tools
    
    async def _get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock data"""
        try:
            # Get data from multiple sources
            yf_data = await self.data_services["yfinance"].get_stock_data(symbol)
            av_data = await self.data_services["alpha_vantage"].get_stock_data(symbol)
            
            return {
                "symbol": symbol,
                "yfinance_data": yf_data,
                "alpha_vantage_data": av_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data"""
        try:
            fundamentals = await self.data_services["yfinance"].get_fundamentals(symbol)
            return {
                "symbol": symbol,
                "fundamentals": fundamentals,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _get_sec_filings(self, symbol: str) -> Dict[str, Any]:
        """Get SEC filings for a company"""
        try:
            filings = await self.data_services["sec_filings"].get_filings(symbol)
            return {
                "symbol": symbol,
                "filings": filings,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting SEC filings for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _get_financial_news(self, query: str) -> Dict[str, Any]:
        """Get financial news and sentiment"""
        try:
            news_data = await self.data_services["news"].get_news(query)
            return {
                "query": query,
                "news_data": news_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting news for {query}: {e}")
            return {"error": str(e), "query": query}
    
    async def _get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Get earnings call data"""
        try:
            earnings_data = await self.data_services["earnings_calls"].get_earnings_data(symbol)
            return {
                "symbol": symbol,
                "earnings_data": earnings_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _get_market_data(self, indicators: List[str] = None) -> Dict[str, Any]:
        """Get market data and economic indicators"""
        try:
            if indicators is None:
                indicators = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
            
            market_data = {}
            for indicator in indicators:
                data = await self.data_services["yfinance"].get_stock_data(indicator)
                market_data[indicator] = data
            
            return {
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    async def execute(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute research task"""
        if not self.is_initialized:
            raise RuntimeError("Research Agent not initialized")
        
        try:
            logger.info(f"Research Agent executing query: {query}")
            
            # Prepare input for the agent
            agent_input = {
                "input": query,
                "parameters": parameters or {}
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Structure the response
            research_results = {
                "agent": self.name,
                "query": query,
                "parameters": parameters,
                "results": result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"Research Agent completed query: {query}")
            return research_results
            
        except Exception as e:
            logger.error(f"Error in Research Agent execution: {e}")
            return {
                "agent": self.name,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            for service in self.data_services.values():
                await service.cleanup()
            logger.info("Research Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during Research Agent cleanup: {e}")