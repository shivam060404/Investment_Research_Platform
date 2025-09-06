from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from ..services.validation_services import (
    FactCheckingService,
    BiasDetectionService,
    DataConsistencyService,
    SourceReliabilityService
)
from ..core.config import settings
from .base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    """Agent responsible for validating analysis results and detecting biases"""
    
    def __init__(self):
        super().__init__("ValidationAgent")
        self.validation_services = {}
        self.llm = None
        self.agent_executor = None
    
    async def initialize(self):
        """Initialize the validation agent and validation services"""
        try:
            logger.info("Initializing Validation Agent")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1  # Low temperature for consistent validation
            )
            
            # Initialize validation services
            self.validation_services = {
                "fact_checking": FactCheckingService(),
                "bias_detection": BiasDetectionService(),
                "data_consistency": DataConsistencyService(),
                "source_reliability": SourceReliabilityService()
            }
            
            # Initialize each service
            for name, service in self.validation_services.items():
                await service.initialize()
                logger.info(f"{name} validation service initialized")
            
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("user", "{input}"),
                ("assistant", "I'll thoroughly validate the analysis results, checking for accuracy, biases, and consistency.")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            self.is_initialized = True
            logger.info("Validation Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Validation Agent: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the validation agent"""
        return """
        You are an expert validation agent responsible for ensuring the accuracy and reliability of investment analysis.
        
        Your capabilities include:
        - Fact-checking claims against reliable financial data sources
        - Detecting cognitive biases and analytical errors
        - Validating data consistency across multiple sources
        - Assessing the reliability and credibility of information sources
        - Identifying potential conflicts of interest or misleading information
        - Cross-referencing analysis results with historical patterns
        
        When validating analysis results, you should:
        1. Verify factual claims against authoritative sources
        2. Check for logical consistency in reasoning
        3. Identify potential biases (confirmation bias, anchoring, etc.)
        4. Assess the quality and reliability of data sources
        5. Look for contradictory evidence or alternative interpretations
        6. Evaluate the statistical significance of findings
        7. Check for proper risk disclosures and limitations
        
        Always provide specific evidence for your validation findings and assign confidence scores to your assessments.
        Flag any areas of concern or uncertainty that require additional investigation.
        """
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the validation agent"""
        tools = [
            Tool(
                name="fact_check_claims",
                description="Verify factual claims against reliable data sources",
                func=self._fact_check_claims
            ),
            Tool(
                name="detect_biases",
                description="Identify potential cognitive biases in analysis",
                func=self._detect_biases
            ),
            Tool(
                name="check_data_consistency",
                description="Validate consistency of data across sources",
                func=self._check_data_consistency
            ),
            Tool(
                name="assess_source_reliability",
                description="Evaluate the reliability and credibility of information sources",
                func=self._assess_source_reliability
            ),
            Tool(
                name="cross_reference_analysis",
                description="Cross-reference analysis with historical data and patterns",
                func=self._cross_reference_analysis
            ),
            Tool(
                name="validate_statistical_claims",
                description="Validate statistical claims and methodologies",
                func=self._validate_statistical_claims
            )
        ]
        return tools
    
    async def _fact_check_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Fact-check specific claims against reliable sources"""
        try:
            results = await self.validation_services["fact_checking"].verify_claims(claims)
            return {
                "validation_type": "fact_checking",
                "claims": claims,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in fact checking: {e}")
            return {"error": str(e), "validation_type": "fact_checking"}
    
    async def _detect_biases(self, analysis_text: str) -> Dict[str, Any]:
        """Detect potential biases in analysis"""
        try:
            results = await self.validation_services["bias_detection"].detect_biases(analysis_text)
            return {
                "validation_type": "bias_detection",
                "analysis_text": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            return {"error": str(e), "validation_type": "bias_detection"}
    
    async def _check_data_consistency(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency across multiple data sources"""
        try:
            results = await self.validation_services["data_consistency"].check_consistency(data_sources)
            return {
                "validation_type": "data_consistency",
                "sources_checked": list(data_sources.keys()),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in data consistency check: {e}")
            return {"error": str(e), "validation_type": "data_consistency"}
    
    async def _assess_source_reliability(self, sources: List[str]) -> Dict[str, Any]:
        """Assess the reliability of information sources"""
        try:
            results = await self.validation_services["source_reliability"].assess_sources(sources)
            return {
                "validation_type": "source_reliability",
                "sources": sources,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in source reliability assessment: {e}")
            return {"error": str(e), "validation_type": "source_reliability"}
    
    async def _cross_reference_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference analysis with historical patterns"""
        try:
            # Extract key metrics and claims from analysis
            key_metrics = self._extract_key_metrics(analysis_results)
            
            # Cross-reference with historical data
            cross_ref_results = await self.validation_services["fact_checking"].cross_reference_historical(key_metrics)
            
            return {
                "validation_type": "cross_reference",
                "key_metrics": key_metrics,
                "results": cross_ref_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in cross-reference analysis: {e}")
            return {"error": str(e), "validation_type": "cross_reference"}
    
    async def _validate_statistical_claims(self, statistical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical claims and methodologies"""
        try:
            results = await self.validation_services["data_consistency"].validate_statistics(statistical_data)
            return {
                "validation_type": "statistical_validation",
                "statistical_data": statistical_data,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in statistical validation: {e}")
            return {"error": str(e), "validation_type": "statistical_validation"}
    
    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics and claims from analysis results"""
        try:
            key_metrics = {
                "financial_ratios": [],
                "price_targets": [],
                "growth_rates": [],
                "risk_metrics": [],
                "market_comparisons": []
            }
            
            # Extract metrics from analysis results
            if "analysis_results" in analysis_results:
                analysis_data = analysis_results["analysis_results"]
                
                # Look for common financial metrics
                for key, value in analysis_data.items():
                    if isinstance(value, dict):
                        if "pe_ratio" in str(value).lower():
                            key_metrics["financial_ratios"].append(value)
                        elif "target" in str(value).lower():
                            key_metrics["price_targets"].append(value)
                        elif "growth" in str(value).lower():
                            key_metrics["growth_rates"].append(value)
                        elif "risk" in str(value).lower():
                            key_metrics["risk_metrics"].append(value)
            
            return key_metrics
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {}
    
    async def execute(self, analysis_results: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute validation task on analysis results"""
        if not self.is_initialized:
            raise RuntimeError("Validation Agent not initialized")
        
        try:
            self._log_execution_start("analysis validation")
            
            # Prepare validation input
            validation_input = {
                "input": f"Validate the following analysis results for accuracy, biases, and consistency: {analysis_results}",
                "analysis_results": analysis_results,
                "parameters": parameters or {}
            }
            
            # Execute the agent
            result = await self.agent_executor.ainvoke(validation_input)
            
            # Perform comprehensive validation
            validation_summary = await self._perform_comprehensive_validation(analysis_results)
            
            # Structure the response
            validation_results = {
                "agent": self.name,
                "analysis_summary": self._summarize_analysis(analysis_results),
                "validation_results": result,
                "comprehensive_validation": validation_summary,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            self._log_execution_end("analysis validation", success=True)
            return validation_results
            
        except Exception as e:
            self._log_execution_end("analysis validation", success=False)
            return self._handle_error(e, "validation execution")
    
    async def _perform_comprehensive_validation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive validation using all validation services"""
        try:
            validation_tasks = []
            
            # Extract text for bias detection
            analysis_text = str(analysis_results.get("analysis_results", ""))
            if analysis_text:
                validation_tasks.append(self._detect_biases(analysis_text))
            
            # Check data consistency if multiple sources present
            if "research_data_summary" in analysis_results:
                data_sources = analysis_results["research_data_summary"]
                validation_tasks.append(self._check_data_consistency(data_sources))
            
            # Cross-reference analysis
            validation_tasks.append(self._cross_reference_analysis(analysis_results))
            
            # Execute all validation tasks
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Compile validation summary
            summary = {
                "total_validations": len(validation_results),
                "successful_validations": sum(1 for r in validation_results if not isinstance(r, Exception) and "error" not in r),
                "validation_details": [r for r in validation_results if not isinstance(r, Exception)],
                "validation_errors": [str(r) for r in validation_results if isinstance(r, Exception)],
                "overall_confidence": self._calculate_overall_confidence(validation_results)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_confidence(self, validation_results: List[Any]) -> float:
        """Calculate overall confidence score based on validation results"""
        try:
            confidence_scores = []
            
            for result in validation_results:
                if isinstance(result, dict) and "results" in result:
                    if "confidence_score" in result["results"]:
                        confidence_scores.append(result["results"]["confidence_score"])
                    elif "reliability_score" in result["results"]:
                        confidence_scores.append(result["results"]["reliability_score"])
            
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.5  # Neutral confidence if no scores available
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _summarize_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the analysis results for validation"""
        try:
            return {
                "analysis_agent": analysis_results.get("agent"),
                "analysis_timestamp": analysis_results.get("timestamp"),
                "analysis_status": analysis_results.get("status"),
                "key_findings": self._extract_key_findings(analysis_results),
                "data_sources_used": analysis_results.get("research_data_summary", {}).get("data_sources", [])
            }
        except Exception as e:
            logger.error(f"Error summarizing analysis: {e}")
            return {"error": str(e)}
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        try:
            findings = []
            
            # Look for key findings in the analysis results
            if "analysis_results" in analysis_results:
                analysis_data = analysis_results["analysis_results"]
                
                # Extract findings based on common patterns
                if isinstance(analysis_data, dict):
                    for key, value in analysis_data.items():
                        if "recommendation" in key.lower():
                            findings.append(f"Recommendation: {value}")
                        elif "conclusion" in key.lower():
                            findings.append(f"Conclusion: {value}")
                        elif "target" in key.lower():
                            findings.append(f"Target: {value}")
            
            return findings[:5]  # Limit to top 5 findings
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            for service in self.validation_services.values():
                await service.cleanup()
            logger.info("Validation Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during Validation Agent cleanup: {e}")