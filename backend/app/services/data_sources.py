from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from loguru import logger

class BaseDataService(ABC):
    """Base class for all data services"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.session = None
    
    async def initialize(self):
        """Initialize the data service"""
        self.session = aiohttp.ClientSession()
        self.is_initialized = True
        logger.info(f"{self.name} data service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info(f"{self.name} data service cleaned up")
    
    @abstractmethod
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get data for a symbol"""
        pass

class YFinanceService(BaseDataService):
    """Yahoo Finance data service"""
    
    def __init__(self):
        super().__init__("YFinance")
    
    async def get_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Get stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            
            # Get info
            info = ticker.info
            
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            return {
                "symbol": symbol,
                "historical_data": hist.to_dict() if not hist.empty else {},
                "info": info,
                "financials": financials.to_dict() if not financials.empty else {},
                "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                "cashflow": cashflow.to_dict() if not cashflow.empty else {},
                "timestamp": datetime.now().isoformat(),
                "source": "yahoo_finance"
            }
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "yahoo_finance"}
    
    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "enterprise_value": info.get("enterpriseValue"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_assets": info.get("returnOnAssets"),
                "return_on_equity": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "analyst_target_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationMean")
            }
            
            return {
                "symbol": symbol,
                "fundamentals": fundamentals,
                "timestamp": datetime.now().isoformat(),
                "source": "yahoo_finance"
            }
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "yahoo_finance"}
    
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get comprehensive data for a symbol"""
        stock_data = await self.get_stock_data(symbol, kwargs.get("period", "1y"))
        fundamentals = await self.get_fundamentals(symbol)
        
        return {
            "stock_data": stock_data,
            "fundamentals": fundamentals
        }

class AlphaVantageService(BaseDataService):
    """Alpha Vantage data service"""
    
    def __init__(self, api_key: str):
        super().__init__("AlphaVantage")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    async def get_stock_data(self, symbol: str, function: str = "TIME_SERIES_DAILY") -> Dict[str, Any]:
        """Get stock data from Alpha Vantage"""
        if not self.api_key:
            return {"error": "Alpha Vantage API key not provided", "symbol": symbol}
        
        try:
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "compact"
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                if "Error Message" in data:
                    return {"error": data["Error Message"], "symbol": symbol}
                
                return {
                    "symbol": symbol,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "source": "alpha_vantage"
                }
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "alpha_vantage"}
    
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage"""
        if not self.api_key:
            return {"error": "Alpha Vantage API key not provided", "symbol": symbol}
        
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                return {
                    "symbol": symbol,
                    "overview": data,
                    "timestamp": datetime.now().isoformat(),
                    "source": "alpha_vantage"
                }
        except Exception as e:
            logger.error(f"Error getting company overview for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "alpha_vantage"}
    
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get comprehensive data for a symbol"""
        stock_data = await self.get_stock_data(symbol)
        overview = await self.get_company_overview(symbol)
        
        return {
            "stock_data": stock_data,
            "overview": overview
        }

class SECFilingsService(BaseDataService):
    """SEC filings data service"""
    
    def __init__(self):
        super().__init__("SECFilings")
        self.base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
    
    async def get_filings(self, symbol: str) -> Dict[str, Any]:
        """Get SEC filings for a company"""
        try:

            return {
                "symbol": symbol,
                "filings": {
                    "10-K": [],
                    "10-Q": [],
                    "8-K": []
                },
                "timestamp": datetime.now().isoformat(),
                "source": "sec_edgar",
                "note": "This is a placeholder implementation"
            }
        except Exception as e:
            logger.error(f"Error getting SEC filings for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "sec_edgar"}
    
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get SEC filing data for a symbol"""
        return await self.get_filings(symbol)

class NewsService(BaseDataService):
    """Financial news data service"""
    
    def __init__(self):
        super().__init__("News")
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.reuters.com/reuters/businessNews"
        ]
    
    async def get_news(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Get financial news related to query"""
        try:

            news_articles = [
                {
                    "title": f"Sample news article about {query}",
                    "summary": f"This is a sample news summary related to {query}",
                    "url": "https://example.com/news/1",
                    "published_date": datetime.now().isoformat(),
                    "source": "Sample News Source",
                    "sentiment": "neutral"
                }
            ]
            
            return {
                "query": query,
                "articles": news_articles,
                "total_articles": len(news_articles),
                "timestamp": datetime.now().isoformat(),
                "source": "news_aggregator",
                "note": "This is a placeholder implementation"
            }
        except Exception as e:
            logger.error(f"Error getting news for {query}: {e}")
            return {"error": str(e), "query": query, "source": "news_aggregator"}
    
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get news data for a symbol"""
        return await self.get_news(symbol)

class EarningsCallService(BaseDataService):
    """Earnings call data service"""
    
    def __init__(self):
        super().__init__("EarningsCalls")
    
    async def get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Get earnings call data for a company"""
        try:

            earnings_data = {
                "latest_call": {
                    "date": datetime.now().isoformat(),
                    "quarter": "Q4 2024",
                    "transcript_summary": f"Summary of {symbol} earnings call",
                    "key_metrics": {
                        "revenue": "$X.X billion",
                        "eps": "$X.XX",
                        "guidance": "Positive outlook for next quarter"
                    },
                    "management_tone": "optimistic",
                    "analyst_questions": [
                        "Question about future growth prospects",
                        "Question about market competition"
                    ]
                },
                "historical_calls": []
            }
            
            return {
                "symbol": symbol,
                "earnings_data": earnings_data,
                "timestamp": datetime.now().isoformat(),
                "source": "earnings_calls",
                "note": "This is a placeholder implementation"
            }
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol, "source": "earnings_calls"}
    
    async def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get earnings data for a symbol"""
        return await self.get_earnings_data(symbol)

# Data aggregator service
class DataAggregatorService:
    """Service to aggregate data from multiple sources"""
    
    def __init__(self):
        self.services = {}
    
    def add_service(self, name: str, service: BaseDataService):
        """Add a data service"""
        self.services[name] = service
    
    async def get_comprehensive_data(self, symbol: str, sources: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive data from multiple sources"""
        if sources is None:
            sources = list(self.services.keys())
        
        results = {}
        tasks = []
        
        for source in sources:
            if source in self.services:
                task = self.services[source].get_data(symbol)
                tasks.append((source, task))
        

        for source, task in tasks:
            try:
                result = await task
                results[source] = result
            except Exception as e:
                logger.error(f"Error getting data from {source}: {e}")
                results[source] = {"error": str(e), "source": source}
        
        return {
            "symbol": symbol,
            "sources": results,
            "timestamp": datetime.now().isoformat(),
            "aggregated_by": "data_aggregator"
        }
    
    async def initialize_all(self):
        """Initialize all data services"""
        for service in self.services.values():
            await service.initialize()
    
    async def cleanup_all(self):
        """Cleanup all data services"""
        for service in self.services.values():
            await service.cleanup()