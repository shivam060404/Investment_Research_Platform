from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    """Status enumeration for responses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentStatusEnum(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

class ConfidenceLevelEnum(str, Enum):
    """Confidence level enumeration"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Base response model
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    request_id: str = Field(..., description="Request identifier")
    status: StatusEnum = Field(..., description="Response status")
    timestamp: datetime = Field(..., description="Response timestamp")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    errors: Optional[List[str]] = Field(default=None, description="List of errors if any")
    warnings: Optional[List[str]] = Field(default=None, description="List of warnings if any")

# Research response models
class ResearchDataPoint(BaseModel):
    """Individual research data point"""
    source: str = Field(..., description="Data source name")
    symbol: Optional[str] = Field(default=None, description="Financial symbol")
    data_type: str = Field(..., description="Type of data")
    value: Any = Field(..., description="Data value")
    timestamp: datetime = Field(..., description="Data timestamp")
    confidence: float = Field(..., description="Data confidence score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ResearchSummary(BaseModel):
    """Summary of research results"""
    total_sources: int = Field(..., description="Total number of data sources")
    successful_sources: int = Field(..., description="Number of successful data retrievals")
    failed_sources: int = Field(..., description="Number of failed data retrievals")
    data_quality_score: float = Field(..., description="Overall data quality score")
    coverage_score: float = Field(..., description="Data coverage score")
    symbols_analyzed: List[str] = Field(..., description="List of symbols analyzed")
    timeframe_covered: str = Field(..., description="Timeframe of data coverage")

class ResearchResponse(BaseResponse):
    """Response model for research operations"""
    query: str = Field(..., description="Original research query")
    results: Dict[str, Any] = Field(..., description="Research results")
    data_points: Optional[List[ResearchDataPoint]] = Field(default=None, description="Individual data points")
    summary: Optional[ResearchSummary] = Field(default=None, description="Research summary")
    sources_used: List[str] = Field(..., description="Data sources used")
    symbols_found: List[str] = Field(..., description="Symbols found in research")

# Analysis response models
class AnalysisResult(BaseModel):
    """Individual analysis result"""
    analysis_type: str = Field(..., description="Type of analysis performed")
    symbol: Optional[str] = Field(default=None, description="Symbol analyzed")
    metrics: Dict[str, float] = Field(..., description="Analysis metrics")
    insights: List[str] = Field(..., description="Key insights")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    supporting_data: Optional[Dict[str, Any]] = Field(default=None, description="Supporting data")
    charts: Optional[List[str]] = Field(default=None, description="Chart URLs or data")

class TechnicalAnalysisResult(AnalysisResult):
    """Technical analysis specific result"""
    indicators: Dict[str, float] = Field(..., description="Technical indicators")
    signals: List[str] = Field(..., description="Trading signals")
    support_levels: List[float] = Field(..., description="Support levels")
    resistance_levels: List[float] = Field(..., description="Resistance levels")
    trend_direction: str = Field(..., description="Overall trend direction")
    momentum: str = Field(..., description="Momentum assessment")

class FundamentalAnalysisResult(AnalysisResult):
    """Fundamental analysis specific result"""
    financial_ratios: Dict[str, float] = Field(..., description="Financial ratios")
    valuation_metrics: Dict[str, float] = Field(..., description="Valuation metrics")
    growth_metrics: Dict[str, float] = Field(..., description="Growth metrics")
    profitability_metrics: Dict[str, float] = Field(..., description="Profitability metrics")
    peer_comparison: Optional[Dict[str, Any]] = Field(default=None, description="Peer comparison")
    fair_value_estimate: Optional[float] = Field(default=None, description="Fair value estimate")

class SentimentAnalysisResult(AnalysisResult):
    """Sentiment analysis specific result"""
    overall_sentiment: str = Field(..., description="Overall sentiment")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    news_sentiment: float = Field(..., description="News sentiment score")
    social_sentiment: Optional[float] = Field(default=None, description="Social media sentiment")
    analyst_sentiment: Optional[float] = Field(default=None, description="Analyst sentiment")
    sentiment_trend: str = Field(..., description="Sentiment trend")

class RiskAnalysisResult(AnalysisResult):
    """Risk analysis specific result"""
    risk_score: float = Field(..., description="Overall risk score")
    volatility: float = Field(..., description="Volatility measure")
    var_95: float = Field(..., description="Value at Risk (95%)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    beta: float = Field(..., description="Beta coefficient")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    risk_mitigation: List[str] = Field(..., description="Risk mitigation suggestions")

class AnalysisResponse(BaseResponse):
    """Response model for analysis operations"""
    research_data_id: str = Field(..., description="ID of research data analyzed")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    analysis_results: List[AnalysisResult] = Field(..., description="Detailed analysis results")
    technical_analysis: Optional[TechnicalAnalysisResult] = Field(default=None, description="Technical analysis")
    fundamental_analysis: Optional[FundamentalAnalysisResult] = Field(default=None, description="Fundamental analysis")
    sentiment_analysis: Optional[SentimentAnalysisResult] = Field(default=None, description="Sentiment analysis")
    risk_analysis: Optional[RiskAnalysisResult] = Field(default=None, description="Risk analysis")
    overall_assessment: str = Field(..., description="Overall investment assessment")
    confidence_score: float = Field(..., description="Overall confidence score")

# Validation response models
class ValidationResult(BaseModel):
    """Individual validation result"""
    validation_type: str = Field(..., description="Type of validation performed")
    passed: bool = Field(..., description="Whether validation passed")
    confidence: float = Field(..., description="Validation confidence")
    issues_found: List[str] = Field(..., description="Issues identified")
    recommendations: List[str] = Field(..., description="Recommendations")
    supporting_evidence: Optional[Dict[str, Any]] = Field(default=None, description="Supporting evidence")

class FactCheckResult(ValidationResult):
    """Fact checking specific result"""
    claims_verified: int = Field(..., description="Number of claims verified")
    claims_disputed: int = Field(..., description="Number of claims disputed")
    sources_consulted: List[str] = Field(..., description="Sources consulted")
    disputed_claims: List[str] = Field(..., description="List of disputed claims")

class BiasDetectionResult(ValidationResult):
    """Bias detection specific result"""
    biases_detected: List[str] = Field(..., description="Types of biases detected")
    bias_severity: str = Field(..., description="Overall bias severity")
    affected_sections: List[str] = Field(..., description="Sections affected by bias")
    mitigation_suggestions: List[str] = Field(..., description="Bias mitigation suggestions")

class ValidationResponse(BaseResponse):
    """Response model for validation operations"""
    analysis_data_id: str = Field(..., description="ID of analysis data validated")
    results: Dict[str, Any] = Field(..., description="Validation results")
    validation_results: List[ValidationResult] = Field(..., description="Detailed validation results")
    fact_check_result: Optional[FactCheckResult] = Field(default=None, description="Fact checking results")
    bias_detection_result: Optional[BiasDetectionResult] = Field(default=None, description="Bias detection results")
    overall_validity: str = Field(..., description="Overall validity assessment")
    reliability_score: float = Field(..., description="Overall reliability score")
    recommendations: List[str] = Field(..., description="Validation recommendations")

# Strategy response models
class StrategyComponent(BaseModel):
    """Individual strategy component"""
    component_type: str = Field(..., description="Type of strategy component")
    recommendations: List[str] = Field(..., description="Component recommendations")
    parameters: Dict[str, Any] = Field(..., description="Component parameters")
    expected_outcome: str = Field(..., description="Expected outcome")
    risk_level: str = Field(..., description="Risk level")
    confidence: float = Field(..., description="Confidence in component")

class PortfolioAllocation(BaseModel):
    """Portfolio allocation recommendation"""
    asset: str = Field(..., description="Asset symbol")
    allocation_percentage: float = Field(..., description="Recommended allocation percentage")
    position_size: float = Field(..., description="Position size in dollars")
    rationale: str = Field(..., description="Rationale for allocation")
    risk_contribution: float = Field(..., description="Risk contribution")

class TradingStrategy(BaseModel):
    """Trading strategy recommendation"""
    strategy_name: str = Field(..., description="Strategy name")
    entry_criteria: List[str] = Field(..., description="Entry criteria")
    exit_criteria: List[str] = Field(..., description="Exit criteria")
    position_sizing: str = Field(..., description="Position sizing method")
    risk_management: List[str] = Field(..., description="Risk management rules")
    expected_return: float = Field(..., description="Expected annual return")
    max_drawdown: float = Field(..., description="Maximum expected drawdown")

class MonitoringPlan(BaseModel):
    """Monitoring plan details"""
    monitoring_frequency: str = Field(..., description="Monitoring frequency")
    key_metrics: List[str] = Field(..., description="Key metrics to monitor")
    alert_thresholds: Dict[str, float] = Field(..., description="Alert thresholds")
    rebalancing_triggers: List[str] = Field(..., description="Rebalancing triggers")
    review_schedule: Dict[str, str] = Field(..., description="Review schedule")

class StrategyResponse(BaseResponse):
    """Response model for strategy operations"""
    validation_data_id: str = Field(..., description="ID of validation data used")
    results: Dict[str, Any] = Field(..., description="Strategy results")
    strategy_components: List[StrategyComponent] = Field(..., description="Strategy components")
    portfolio_allocation: List[PortfolioAllocation] = Field(..., description="Portfolio allocation")
    trading_strategy: Optional[TradingStrategy] = Field(default=None, description="Trading strategy")
    monitoring_plan: MonitoringPlan = Field(..., description="Monitoring plan")
    overall_recommendation: str = Field(..., description="Overall investment recommendation")
    expected_return: float = Field(..., description="Expected annual return")
    risk_assessment: str = Field(..., description="Risk assessment")
    implementation_timeline: Dict[str, List[str]] = Field(..., description="Implementation timeline")

# Monitoring response models
class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")

class AlertInfo(BaseModel):
    """Alert information"""
    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    triggered_at: datetime = Field(..., description="When alert was triggered")
    metric_value: float = Field(..., description="Metric value that triggered alert")
    threshold_value: float = Field(..., description="Threshold that was breached")

class MonitoringResponse(BaseResponse):
    """Response model for monitoring operations"""
    workflow_id: str = Field(..., description="Workflow ID being monitored")
    monitoring_status: str = Field(..., description="Current monitoring status")
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None, description="Performance metrics")
    active_alerts: List[AlertInfo] = Field(..., description="Active alerts")
    last_update: datetime = Field(..., description="Last monitoring update")
    next_review: datetime = Field(..., description="Next scheduled review")

# Workflow response models
class WorkflowStage(BaseModel):
    """Individual workflow stage"""
    stage_name: str = Field(..., description="Stage name")
    status: StatusEnum = Field(..., description="Stage status")
    start_time: Optional[datetime] = Field(default=None, description="Stage start time")
    end_time: Optional[datetime] = Field(default=None, description="Stage end time")
    duration_ms: Optional[float] = Field(default=None, description="Stage duration in milliseconds")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Stage output")
    errors: Optional[List[str]] = Field(default=None, description="Stage errors")

class WorkflowResponse(BaseResponse):
    """Response model for complete workflow operations"""
    workflow_id: str = Field(..., description="Workflow identifier")
    query: str = Field(..., description="Original query")
    results: Dict[str, Any] = Field(..., description="Complete workflow results")
    stages: List[WorkflowStage] = Field(..., description="Workflow stages")
    research_summary: Optional[ResearchSummary] = Field(default=None, description="Research summary")
    analysis_summary: Optional[str] = Field(default=None, description="Analysis summary")
    validation_summary: Optional[str] = Field(default=None, description="Validation summary")
    strategy_summary: Optional[str] = Field(default=None, description="Strategy summary")
    final_recommendation: str = Field(..., description="Final investment recommendation")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Overall confidence level")
    monitoring_enabled: bool = Field(..., description="Whether monitoring is enabled")

# System response models
class AgentStatus(BaseModel):
    """Agent status information"""
    agent_name: str = Field(..., description="Agent name")
    status: AgentStatusEnum = Field(..., description="Agent status")
    last_execution: Optional[datetime] = Field(default=None, description="Last execution time")
    execution_count: int = Field(..., description="Total execution count")
    error_count: int = Field(..., description="Total error count")
    average_execution_time: Optional[float] = Field(default=None, description="Average execution time")

class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    active_connections: int = Field(..., description="Active database connections")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    average_response_time: float = Field(..., description="Average response time")

class StatusResponse(BaseResponse):
    """Response model for status operations"""
    service_status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Service version")
    uptime: str = Field(..., description="Service uptime")
    agents: Dict[str, AgentStatusEnum] = Field(..., description="Agent statuses")
    agent_details: Optional[List[AgentStatus]] = Field(default=None, description="Detailed agent information")
    system_metrics: Optional[SystemMetrics] = Field(default=None, description="System metrics")
    database_status: str = Field(..., description="Database connection status")
    external_services: Dict[str, str] = Field(..., description="External service statuses")

# Error response models
class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Error context")

class ErrorResponse(BaseModel):
    """Response model for errors"""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Error timestamp")
    status_code: int = Field(..., description="HTTP status code")
    error: ErrorDetail = Field(..., description="Error details")
    suggestions: Optional[List[str]] = Field(default=None, description="Suggestions to resolve error")

# Pagination response models
class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

class PaginatedResponse(BaseModel):
    """Base paginated response"""
    data: List[Any] = Field(..., description="Response data")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    timestamp: datetime = Field(..., description="Response timestamp")

# Bulk operation response models
class BulkOperationResult(BaseModel):
    """Result of a bulk operation"""
    operation_id: str = Field(..., description="Operation identifier")
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    errors: List[ErrorDetail] = Field(..., description="Operation errors")

class BulkResponse(BaseResponse):
    """Response model for bulk operations"""
    bulk_result: BulkOperationResult = Field(..., description="Bulk operation result")
    summary: str = Field(..., description="Operation summary")