from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

class TimeframeEnum(str, Enum):
    """Supported timeframes for data requests"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class DataSourceEnum(str, Enum):
    """Supported data sources"""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    SEC_FILINGS = "sec_filings"
    NEWS = "news"
    EARNINGS_CALLS = "earnings_calls"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"

class AnalysisTypeEnum(str, Enum):
    """Supported analysis types"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    COMPARATIVE = "comparative"
    SCENARIO = "scenario"

class ValidationTypeEnum(str, Enum):
    """Supported validation types"""
    FACT_CHECKING = "fact_checking"
    BIAS_DETECTION = "bias_detection"
    DATA_CONSISTENCY = "data_consistency"
    SOURCE_RELIABILITY = "source_reliability"
    CROSS_REFERENCE = "cross_reference"
    STATISTICAL_VALIDATION = "statistical_validation"

class StrategyTypeEnum(str, Enum):
    """Supported strategy types"""
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    TRADING_STRATEGY = "trading_strategy"
    RISK_MANAGEMENT = "risk_management"
    ASSET_ALLOCATION = "asset_allocation"
    POSITION_SIZING = "position_sizing"
    MONITORING_PLAN = "monitoring_plan"

class InvestmentHorizonEnum(str, Enum):
    """Investment horizon options"""
    SHORT_TERM = "short_term"  # < 1 year
    MEDIUM_TERM = "medium_term"  # 1-5 years
    LONG_TERM = "long_term"  # > 5 years

class RiskToleranceEnum(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

# Base request model
class BaseRequest(BaseModel):
    """Base request model with common fields"""
    request_id: str = Field(..., description="Unique identifier for the request")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters")

# Research request models
class ResearchRequest(BaseRequest):
    """Request model for research operations"""
    query: str = Field(..., description="Research query or question")
    symbols: Optional[List[str]] = Field(default=None, description="Financial symbols to research")
    data_sources: Optional[List[DataSourceEnum]] = Field(default=None, description="Data sources to use")
    timeframe: Optional[TimeframeEnum] = Field(default=TimeframeEnum.DAILY, description="Data timeframe")
    start_date: Optional[datetime] = Field(default=None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(default=None, description="End date for historical data")
    include_news: bool = Field(default=True, description="Include news data")
    include_filings: bool = Field(default=True, description="Include SEC filings")
    include_earnings: bool = Field(default=True, description="Include earnings data")

# Analysis request models
class AnalysisRequest(BaseRequest):
    """Request model for analysis operations"""
    research_data_id: str = Field(..., description="ID of the research data to analyze")
    research_data: Dict[str, Any] = Field(..., description="Research data to analyze")
    analysis_types: List[AnalysisTypeEnum] = Field(..., description="Types of analysis to perform")
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to focus analysis on")
    benchmark: Optional[str] = Field(default="^GSPC", description="Benchmark for comparison")
    peers: Optional[List[str]] = Field(default=None, description="Peer companies for comparison")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")

# Validation request models
class ValidationRequest(BaseRequest):
    """Request model for validation operations"""
    analysis_data_id: str = Field(..., description="ID of the analysis data to validate")
    analysis_data: Dict[str, Any] = Field(..., description="Analysis data to validate")
    validation_types: List[ValidationTypeEnum] = Field(..., description="Types of validation to perform")
    fact_check_sources: Optional[List[str]] = Field(default=None, description="Sources for fact checking")
    bias_detection_sensitivity: float = Field(default=0.5, description="Sensitivity for bias detection")
    consistency_threshold: float = Field(default=0.8, description="Threshold for data consistency")

# Strategy request models
class StrategyRequest(BaseRequest):
    """Request model for strategy generation"""
    validation_data_id: str = Field(..., description="ID of the validation data")
    validation_data: Dict[str, Any] = Field(..., description="Validated analysis data")
    strategy_types: List[StrategyTypeEnum] = Field(..., description="Types of strategies to generate")
    risk_tolerance: RiskToleranceEnum = Field(default=RiskToleranceEnum.MODERATE, description="Risk tolerance level")
    investment_horizon: InvestmentHorizonEnum = Field(default=InvestmentHorizonEnum.MEDIUM_TERM, description="Investment horizon")
    portfolio_value: float = Field(default=100000, description="Total portfolio value")
    max_position_size: float = Field(default=0.1, description="Maximum position size as percentage")
    diversification_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Diversification constraints")
    target_return: Optional[float] = Field(default=None, description="Target annual return")
    max_drawdown: Optional[float] = Field(default=0.15, description="Maximum acceptable drawdown")

# Monitoring request models
class MonitoringRequest(BaseRequest):
    """Request model for monitoring operations"""
    workflow_id: str = Field(..., description="Workflow ID to monitor")
    strategy_data: Dict[str, Any] = Field(..., description="Strategy data to monitor")
    monitoring_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Monitoring parameters")
    alert_thresholds: Optional[Dict[str, float]] = Field(default=None, description="Custom alert thresholds")
    monitoring_frequency: int = Field(default=300, description="Monitoring frequency in seconds")
    auto_rebalance: bool = Field(default=False, description="Enable automatic rebalancing")
    notification_channels: Optional[List[str]] = Field(default=None, description="Notification channels")

# Workflow request models
class WorkflowRequest(BaseRequest):
    """Request model for complete workflow execution"""
    query: str = Field(..., description="Investment research query")
    symbols: Optional[List[str]] = Field(default=None, description="Financial symbols to research")
    data_sources: Optional[List[DataSourceEnum]] = Field(default=None, description="Data sources to use")
    analysis_types: Optional[List[AnalysisTypeEnum]] = Field(default=None, description="Analysis types to perform")
    validation_types: Optional[List[ValidationTypeEnum]] = Field(default=None, description="Validation types to perform")
    strategy_types: Optional[List[StrategyTypeEnum]] = Field(default=None, description="Strategy types to generate")
    risk_tolerance: RiskToleranceEnum = Field(default=RiskToleranceEnum.MODERATE, description="Risk tolerance")
    investment_horizon: InvestmentHorizonEnum = Field(default=InvestmentHorizonEnum.MEDIUM_TERM, description="Investment horizon")
    portfolio_value: float = Field(default=100000, description="Portfolio value")
    enable_monitoring: bool = Field(default=True, description="Enable continuous monitoring")
    timeframe: Optional[TimeframeEnum] = Field(default=TimeframeEnum.DAILY, description="Data timeframe")

# Specialized request models
class BacktestRequest(BaseRequest):
    """Request model for backtesting strategies"""
    strategy_data: Dict[str, Any] = Field(..., description="Strategy to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000, description="Initial capital for backtest")
    benchmark: str = Field(default="^GSPC", description="Benchmark for comparison")
    transaction_costs: float = Field(default=0.001, description="Transaction costs as percentage")
    rebalancing_frequency: str = Field(default="monthly", description="Rebalancing frequency")

class OptimizationRequest(BaseRequest):
    """Request model for portfolio optimization"""
    assets: List[str] = Field(..., description="Assets to include in optimization")
    expected_returns: Optional[Dict[str, float]] = Field(default=None, description="Expected returns for assets")
    risk_model: Optional[Dict[str, Any]] = Field(default=None, description="Risk model parameters")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Optimization constraints")
    objective: str = Field(default="max_sharpe", description="Optimization objective")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")

class AlertRequest(BaseRequest):
    """Request model for setting up alerts"""
    workflow_id: str = Field(..., description="Workflow ID to set alerts for")
    alert_conditions: List[Dict[str, Any]] = Field(..., description="Alert conditions")
    notification_methods: List[str] = Field(..., description="Notification methods")
    priority_level: str = Field(default="medium", description="Alert priority level")
    active: bool = Field(default=True, description="Whether alert is active")

# Bulk operation request models
class BulkResearchRequest(BaseRequest):
    """Request model for bulk research operations"""
    queries: List[str] = Field(..., description="Multiple research queries")
    symbols_list: Optional[List[List[str]]] = Field(default=None, description="Symbols for each query")
    common_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Common parameters for all queries")
    parallel_execution: bool = Field(default=True, description="Execute queries in parallel")
    max_concurrent: int = Field(default=5, description="Maximum concurrent executions")

class BulkAnalysisRequest(BaseRequest):
    """Request model for bulk analysis operations"""
    research_data_list: List[Dict[str, Any]] = Field(..., description="Multiple research datasets")
    analysis_types: List[AnalysisTypeEnum] = Field(..., description="Analysis types to perform")
    common_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Common parameters")
    parallel_execution: bool = Field(default=True, description="Execute analyses in parallel")

# Configuration request models
class ConfigurationRequest(BaseRequest):
    """Request model for system configuration"""
    agent_config: Optional[Dict[str, Any]] = Field(default=None, description="Agent configuration")
    data_source_config: Optional[Dict[str, Any]] = Field(default=None, description="Data source configuration")
    monitoring_config: Optional[Dict[str, Any]] = Field(default=None, description="Monitoring configuration")
    notification_config: Optional[Dict[str, Any]] = Field(default=None, description="Notification configuration")
    apply_immediately: bool = Field(default=False, description="Apply configuration immediately")

class HealthCheckRequest(BaseRequest):
    """Request model for health checks"""
    include_agents: bool = Field(default=True, description="Include agent status")
    include_services: bool = Field(default=True, description="Include service status")
    include_performance: bool = Field(default=False, description="Include performance metrics")
    detailed: bool = Field(default=False, description="Include detailed diagnostics")