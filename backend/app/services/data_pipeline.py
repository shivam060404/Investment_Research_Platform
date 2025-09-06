from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from loguru import logger
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import websockets
from redis import Redis
from sqlalchemy.orm import Session

from ..core.database import get_redis, SessionLocal
from ..core.config import settings

class DataSourceType(Enum):
    """Types of data sources"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    API = "api"
    WEBSOCKET = "websocket"

class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class DataPoint:
    """Represents a single data point in the pipeline"""
    source: str
    symbol: str
    data_type: str
    value: Any
    timestamp: datetime
    quality: DataQuality
    metadata: Dict[str, Any]

class BaseDataIngester(ABC):
    """Base class for data ingesters"""
    
    def __init__(self, name: str, source_type: DataSourceType):
        self.name = name
        self.source_type = source_type
        self.is_running = False
        self.error_count = 0
        self.data_count = 0
        self.last_update = None
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[DataPoint], None]):
        """Add callback for processed data"""
        self.callbacks.append(callback)
    
    async def emit_data(self, data_point: DataPoint):
        """Emit data to all registered callbacks"""
        for callback in self.callbacks:
            try:
                await callback(data_point)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
    
    @abstractmethod
    async def start_ingestion(self):
        """Start data ingestion"""
        pass
    
    @abstractmethod
    async def stop_ingestion(self):
        """Stop data ingestion"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get ingester status"""
        pass

class RealTimeMarketDataIngester(BaseDataIngester):
    """Real-time market data ingester using WebSocket connections"""
    
    def __init__(self, symbols: List[str]):
        super().__init__("RealTimeMarketData", DataSourceType.WEBSOCKET)
        self.symbols = symbols
        self.websocket = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
    
    async def start_ingestion(self):
        """Start real-time market data ingestion"""
        self.is_running = True
        logger.info(f"Starting real-time market data ingestion for {len(self.symbols)} symbols")
        
        while self.is_running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"Error in market data ingestion: {e}")
                self.error_count += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    wait_time = min(60, 2 ** self.reconnect_attempts)
                    logger.info(f"Reconnecting in {wait_time} seconds (attempt {self.reconnect_attempts})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max reconnection attempts reached, stopping ingestion")
                    break
    
    async def _connect_and_stream(self):
        """Connect to WebSocket and stream data"""
        # Mock WebSocket connection for demonstration
        # In real implementation, connect to actual market data providers
        
        for symbol in self.symbols:
            # Simulate real-time price updates
            price = 100 + np.random.normal(0, 2)  # Mock price with volatility
            volume = np.random.randint(1000, 10000)
            
            data_point = DataPoint(
                source="mock_market_data",
                symbol=symbol,
                data_type="price_update",
                value={
                    "price": price,
                    "volume": volume,
                    "bid": price - 0.01,
                    "ask": price + 0.01
                },
                timestamp=datetime.now(),
                quality=DataQuality.HIGH,
                metadata={"exchange": "NASDAQ", "currency": "USD"}
            )
            
            await self.emit_data(data_point)
            self.data_count += 1
            self.last_update = datetime.now()
        
        # Simulate streaming delay
        await asyncio.sleep(1)
    
    async def stop_ingestion(self):
        """Stop real-time market data ingestion"""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
        logger.info("Stopped real-time market data ingestion")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get ingester status"""
        return {
            "name": self.name,
            "type": self.source_type.value,
            "is_running": self.is_running,
            "symbols_count": len(self.symbols),
            "data_points_processed": self.data_count,
            "error_count": self.error_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "reconnect_attempts": self.reconnect_attempts
        }

class NewsDataIngester(BaseDataIngester):
    """News data ingester for financial news"""
    
    def __init__(self, news_sources: List[str]):
        super().__init__("NewsData", DataSourceType.API)
        self.news_sources = news_sources
        self.session = None
        self.last_fetch_time = {}
    
    async def start_ingestion(self):
        """Start news data ingestion"""
        self.is_running = True
        self.session = aiohttp.ClientSession()
        logger.info(f"Starting news data ingestion from {len(self.news_sources)} sources")
        
        while self.is_running:
            try:
                await self._fetch_news_data()
                await asyncio.sleep(300)  # Fetch every 5 minutes
            except Exception as e:
                logger.error(f"Error in news data ingestion: {e}")
                self.error_count += 1
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _fetch_news_data(self):
        """Fetch news data from sources"""
        for source in self.news_sources:
            try:
                # Mock news data fetching
                # In real implementation, call actual news APIs
                
                news_articles = [
                    {
                        "title": f"Market Update: {source} reports on market conditions",
                        "content": "Sample news content about market movements and analysis",
                        "url": f"https://{source}/article/123",
                        "published_at": datetime.now().isoformat(),
                        "sentiment": "neutral",
                        "relevance_score": 0.8
                    }
                ]
                
                for article in news_articles:
                    data_point = DataPoint(
                        source=source,
                        symbol="MARKET",
                        data_type="news_article",
                        value=article,
                        timestamp=datetime.now(),
                        quality=DataQuality.MEDIUM,
                        metadata={"source_type": "news", "language": "en"}
                    )
                    
                    await self.emit_data(data_point)
                    self.data_count += 1
                
                self.last_fetch_time[source] = datetime.now()
                self.last_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {e}")
                self.error_count += 1
    
    async def stop_ingestion(self):
        """Stop news data ingestion"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Stopped news data ingestion")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get ingester status"""
        return {
            "name": self.name,
            "type": self.source_type.value,
            "is_running": self.is_running,
            "sources_count": len(self.news_sources),
            "data_points_processed": self.data_count,
            "error_count": self.error_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_fetch_times": {k: v.isoformat() for k, v in self.last_fetch_time.items()}
        }

class EconomicDataIngester(BaseDataIngester):
    """Economic indicators data ingester"""
    
    def __init__(self, indicators: List[str]):
        super().__init__("EconomicData", DataSourceType.BATCH)
        self.indicators = indicators
        self.session = None
    
    async def start_ingestion(self):
        """Start economic data ingestion"""
        self.is_running = True
        self.session = aiohttp.ClientSession()
        logger.info(f"Starting economic data ingestion for {len(self.indicators)} indicators")
        
        while self.is_running:
            try:
                await self._fetch_economic_data()
                await asyncio.sleep(3600)  # Fetch every hour
            except Exception as e:
                logger.error(f"Error in economic data ingestion: {e}")
                self.error_count += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _fetch_economic_data(self):
        """Fetch economic indicators data"""
        for indicator in self.indicators:
            try:
                # Mock economic data
                # In real implementation, fetch from FRED, BLS, etc.
                
                economic_data = {
                    "indicator": indicator,
                    "value": np.random.normal(2.5, 0.5),  # Mock economic value
                    "period": datetime.now().strftime("%Y-%m"),
                    "unit": "percent",
                    "frequency": "monthly"
                }
                
                data_point = DataPoint(
                    source="economic_data_api",
                    symbol=indicator,
                    data_type="economic_indicator",
                    value=economic_data,
                    timestamp=datetime.now(),
                    quality=DataQuality.HIGH,
                    metadata={"data_type": "economic", "frequency": "monthly"}
                )
                
                await self.emit_data(data_point)
                self.data_count += 1
                
            except Exception as e:
                logger.error(f"Error fetching economic data for {indicator}: {e}")
                self.error_count += 1
        
        self.last_update = datetime.now()
    
    async def stop_ingestion(self):
        """Stop economic data ingestion"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Stopped economic data ingestion")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get ingester status"""
        return {
            "name": self.name,
            "type": self.source_type.value,
            "is_running": self.is_running,
            "indicators_count": len(self.indicators),
            "data_points_processed": self.data_count,
            "error_count": self.error_count,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

class DataProcessor:
    """Processes and transforms raw data points"""
    
    def __init__(self):
        self.processors = {
            "price_update": self._process_price_update,
            "news_article": self._process_news_article,
            "economic_indicator": self._process_economic_indicator
        }
    
    async def process_data_point(self, data_point: DataPoint) -> Optional[DataPoint]:
        """Process a data point based on its type"""
        try:
            processor = self.processors.get(data_point.data_type)
            if processor:
                return await processor(data_point)
            else:
                logger.warning(f"No processor found for data type: {data_point.data_type}")
                return data_point
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
            return None
    
    async def _process_price_update(self, data_point: DataPoint) -> DataPoint:
        """Process price update data"""
        value = data_point.value
        
        # Add calculated fields
        if "bid" in value and "ask" in value:
            value["spread"] = value["ask"] - value["bid"]
            value["mid_price"] = (value["ask"] + value["bid"]) / 2
        
        # Add quality assessment
        if value.get("spread", 0) > value.get("price", 0) * 0.01:  # Spread > 1% of price
            data_point.quality = DataQuality.LOW
        
        return data_point
    
    async def _process_news_article(self, data_point: DataPoint) -> DataPoint:
        """Process news article data"""
        value = data_point.value
        
        # Extract keywords
        content = value.get("content", "")
        financial_keywords = [
            "earnings", "revenue", "profit", "loss", "growth", "decline",
            "merger", "acquisition", "dividend", "buyback", "ipo"
        ]
        
        found_keywords = [kw for kw in financial_keywords if kw.lower() in content.lower()]
        value["keywords"] = found_keywords
        value["keyword_count"] = len(found_keywords)
        
        # Assess relevance
        if len(found_keywords) >= 3:
            data_point.quality = DataQuality.HIGH
        elif len(found_keywords) >= 1:
            data_point.quality = DataQuality.MEDIUM
        else:
            data_point.quality = DataQuality.LOW
        
        return data_point
    
    async def _process_economic_indicator(self, data_point: DataPoint) -> DataPoint:
        """Process economic indicator data"""
        value = data_point.value
        
        # Add trend analysis
        current_value = value.get("value", 0)
        
        # Mock historical comparison
        historical_avg = 2.0  # Mock historical average
        value["vs_historical_avg"] = current_value - historical_avg
        value["trend"] = "increasing" if current_value > historical_avg else "decreasing"
        
        return data_point

class DataStorage:
    """Handles data storage to various backends"""
    
    def __init__(self):
        self.redis_client = None
        self.db_session = None
    
    async def initialize(self):
        """Initialize storage connections"""
        try:
            self.redis_client = get_redis()
            logger.info("Data storage initialized")
        except Exception as e:
            logger.error(f"Error initializing data storage: {e}")
    
    async def store_data_point(self, data_point: DataPoint):
        """Store data point to appropriate storage"""
        try:
            # Store in Redis for real-time access
            await self._store_in_redis(data_point)
            
            # Store in database for historical analysis
            await self._store_in_database(data_point)
            
        except Exception as e:
            logger.error(f"Error storing data point: {e}")
    
    async def _store_in_redis(self, data_point: DataPoint):
        """Store data point in Redis"""
        try:
            key = f"{data_point.source}:{data_point.symbol}:{data_point.data_type}"
            value = {
                "value": data_point.value,
                "timestamp": data_point.timestamp.isoformat(),
                "quality": data_point.quality.value,
                "metadata": data_point.metadata
            }
            
            # Store with expiration (24 hours)
            self.redis_client.setex(key, 86400, json.dumps(value, default=str))
            
            # Also add to time series for trending
            ts_key = f"ts:{data_point.source}:{data_point.symbol}"
            self.redis_client.zadd(ts_key, {
                json.dumps(value, default=str): data_point.timestamp.timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error storing in Redis: {e}")
    
    async def _store_in_database(self, data_point: DataPoint):
        """Store data point in database"""
        try:
            # In real implementation, store in PostgreSQL
            # For now, just log the storage
            logger.debug(f"Storing in database: {data_point.source}:{data_point.symbol}")
            
        except Exception as e:
            logger.error(f"Error storing in database: {e}")

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self):
        self.ingesters: List[BaseDataIngester] = []
        self.processor = DataProcessor()
        self.storage = DataStorage()
        self.is_running = False
        self.stats = {
            "total_data_points": 0,
            "processed_data_points": 0,
            "stored_data_points": 0,
            "error_count": 0,
            "start_time": None
        }
    
    def add_ingester(self, ingester: BaseDataIngester):
        """Add data ingester to pipeline"""
        ingester.add_callback(self._process_and_store_data)
        self.ingesters.append(ingester)
        logger.info(f"Added ingester: {ingester.name}")
    
    async def _process_and_store_data(self, data_point: DataPoint):
        """Process and store incoming data point"""
        try:
            self.stats["total_data_points"] += 1
            
            # Process data point
            processed_data = await self.processor.process_data_point(data_point)
            if processed_data:
                self.stats["processed_data_points"] += 1
                
                # Store data point
                await self.storage.store_data_point(processed_data)
                self.stats["stored_data_points"] += 1
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            self.stats["error_count"] += 1
    
    async def start(self):
        """Start the data pipeline"""
        try:
            logger.info("Starting data pipeline")
            self.is_running = True
            self.stats["start_time"] = datetime.now()
            
            # Initialize storage
            await self.storage.initialize()
            
            # Start all ingesters
            ingester_tasks = []
            for ingester in self.ingesters:
                task = asyncio.create_task(ingester.start_ingestion())
                ingester_tasks.append(task)
            
            logger.info(f"Started {len(self.ingesters)} data ingesters")
            
            # Wait for all ingesters to complete (or run indefinitely)
            await asyncio.gather(*ingester_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting data pipeline: {e}")
            raise
    
    async def stop(self):
        """Stop the data pipeline"""
        try:
            logger.info("Stopping data pipeline")
            self.is_running = False
            
            # Stop all ingesters
            for ingester in self.ingesters:
                await ingester.stop_ingestion()
            
            logger.info("Data pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data pipeline: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        ingester_statuses = []
        for ingester in self.ingesters:
            status = await ingester.get_status()
            ingester_statuses.append(status)
        
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "pipeline_status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "statistics": self.stats,
            "ingesters": ingester_statuses,
            "timestamp": datetime.now().isoformat()
        }

# Factory function to create configured pipeline
def create_data_pipeline() -> DataPipeline:
    """Create and configure data pipeline with default ingesters"""
    pipeline = DataPipeline()
    
    # Add market data ingester
    market_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    market_ingester = RealTimeMarketDataIngester(market_symbols)
    pipeline.add_ingester(market_ingester)
    
    # Add news data ingester
    news_sources = ["reuters.com", "bloomberg.com", "cnbc.com"]
    news_ingester = NewsDataIngester(news_sources)
    pipeline.add_ingester(news_ingester)
    
    # Add economic data ingester
    economic_indicators = ["GDP", "CPI", "UNEMPLOYMENT", "INTEREST_RATE"]
    economic_ingester = EconomicDataIngester(economic_indicators)
    pipeline.add_ingester(economic_ingester)
    
    return pipeline