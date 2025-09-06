from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class ResearchData(Base, TimestampMixin):
    """Research data storage"""
    __tablename__ = "research_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    query = Column(Text, nullable=False)
    symbols = Column(JSON, nullable=True)  # List of symbols
    sources_used = Column(JSON, nullable=True)  # List of data sources
    data_points = Column(JSON, nullable=True)  # Research data points
    summary = Column(JSON, nullable=True)  # Research summary
    status = Column(String(50), nullable=False, default="pending")
    execution_time_ms = Column(Float, nullable=True)
    errors = Column(JSON, nullable=True)  # List of errors
    
    # Relationships
    analyses = relationship("AnalysisData", back_populates="research")
    
    __table_args__ = (
        Index('idx_research_status', 'status'),
        Index('idx_research_created', 'created_at'),
    )

class AnalysisData(Base, TimestampMixin):
    """Analysis data storage"""
    __tablename__ = "analysis_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    research_data_id = Column(UUID(as_uuid=True), ForeignKey("research_data.id"), nullable=False)
    analysis_type = Column(String(100), nullable=False)
    symbols = Column(JSON, nullable=True)  # List of symbols analyzed
    results = Column(JSON, nullable=True)  # Analysis results
    technical_analysis = Column(JSON, nullable=True)
    fundamental_analysis = Column(JSON, nullable=True)
    sentiment_analysis = Column(JSON, nullable=True)
    risk_analysis = Column(JSON, nullable=True)
    overall_assessment = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    execution_time_ms = Column(Float, nullable=True)
    errors = Column(JSON, nullable=True)
    
    # Relationships
    research = relationship("ResearchData", back_populates="analyses")
    validations = relationship("ValidationData", back_populates="analysis")
    
    __table_args__ = (
        Index('idx_analysis_status', 'status'),
        Index('idx_analysis_type', 'analysis_type'),
        Index('idx_analysis_created', 'created_at'),
    )

class ValidationData(Base, TimestampMixin):
    """Validation data storage"""
    __tablename__ = "validation_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    analysis_data_id = Column(UUID(as_uuid=True), ForeignKey("analysis_data.id"), nullable=False)
    validation_type = Column(String(100), nullable=False)
    results = Column(JSON, nullable=True)  # Validation results
    fact_check_result = Column(JSON, nullable=True)
    bias_detection_result = Column(JSON, nullable=True)
    overall_validity = Column(String(100), nullable=True)
    reliability_score = Column(Float, nullable=True)
    recommendations = Column(JSON, nullable=True)  # List of recommendations
    status = Column(String(50), nullable=False, default="pending")
    execution_time_ms = Column(Float, nullable=True)
    errors = Column(JSON, nullable=True)
    
    # Relationships
    analysis = relationship("AnalysisData", back_populates="validations")
    strategies = relationship("StrategyData", back_populates="validation")
    
    __table_args__ = (
        Index('idx_validation_status', 'status'),
        Index('idx_validation_type', 'validation_type'),
        Index('idx_validation_created', 'created_at'),
    )

class StrategyData(Base, TimestampMixin):
    """Strategy data storage"""
    __tablename__ = "strategy_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    validation_data_id = Column(UUID(as_uuid=True), ForeignKey("validation_data.id"), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    results = Column(JSON, nullable=True)  # Strategy results
    strategy_components = Column(JSON, nullable=True)
    portfolio_allocation = Column(JSON, nullable=True)
    trading_strategy = Column(JSON, nullable=True)
    monitoring_plan = Column(JSON, nullable=True)
    overall_recommendation = Column(Text, nullable=True)
    expected_return = Column(Float, nullable=True)
    risk_assessment = Column(String(100), nullable=True)
    implementation_timeline = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    execution_time_ms = Column(Float, nullable=True)
    errors = Column(JSON, nullable=True)
    
    # Relationships
    validation = relationship("ValidationData", back_populates="strategies")
    monitoring_records = relationship("MonitoringData", back_populates="strategy")
    
    __table_args__ = (
        Index('idx_strategy_status', 'status'),
        Index('idx_strategy_type', 'strategy_type'),
        Index('idx_strategy_created', 'created_at'),
    )

class MonitoringData(Base, TimestampMixin):
    """Monitoring data storage"""
    __tablename__ = "monitoring_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    strategy_data_id = Column(UUID(as_uuid=True), ForeignKey("strategy_data.id"), nullable=False)
    workflow_id = Column(String(255), nullable=False, index=True)
    monitoring_status = Column(String(50), nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    active_alerts = Column(JSON, nullable=True)
    last_update = Column(DateTime, nullable=False, default=datetime.utcnow)
    next_review = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, default="active")
    
    # Relationships
    strategy = relationship("StrategyData", back_populates="monitoring_records")
    
    __table_args__ = (
        Index('idx_monitoring_status', 'monitoring_status'),
        Index('idx_monitoring_workflow', 'workflow_id'),
        Index('idx_monitoring_created', 'created_at'),
    )

class WorkflowExecution(Base, TimestampMixin):
    """Workflow execution tracking"""
    __tablename__ = "workflow_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(String(255), nullable=False, unique=True, index=True)
    query = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    current_stage = Column(String(100), nullable=True)
    stages = Column(JSON, nullable=True)  # List of workflow stages
    results = Column(JSON, nullable=True)  # Complete workflow results
    final_recommendation = Column(Text, nullable=True)
    confidence_level = Column(String(50), nullable=True)
    monitoring_enabled = Column(Boolean, default=False)
    total_execution_time_ms = Column(Float, nullable=True)
    errors = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_workflow_status', 'status'),
        Index('idx_workflow_stage', 'current_stage'),
        Index('idx_workflow_created', 'created_at'),
    )

class SystemMetrics(Base, TimestampMixin):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(100), nullable=False)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_metrics_type', 'metric_type'),
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_created', 'created_at'),
    )

class AgentExecutionLog(Base, TimestampMixin):
    """Agent execution logging"""
    __tablename__ = "agent_execution_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_name = Column(String(100), nullable=False)
    execution_id = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_agent_name', 'agent_name'),
        Index('idx_agent_status', 'status'),
        Index('idx_agent_created', 'created_at'),
    )

class UserSession(Base, TimestampMixin):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=True)  # For future authentication
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_session_active', 'is_active'),
        Index('idx_session_activity', 'last_activity'),
    )