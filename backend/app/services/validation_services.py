from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import re
from loguru import logger
from textblob import TextBlob
import numpy as np

class BaseValidationService(ABC):
    """Base class for all validation services"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the validation service"""
        self.is_initialized = True
        logger.info(f"{self.name} validation service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} validation service cleaned up")
    
    @abstractmethod
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation on the provided data"""
        pass

class FactCheckingService(BaseValidationService):
    """Service for fact-checking financial claims and data"""
    
    def __init__(self):
        super().__init__("FactChecking")
        self.reliable_sources = [
            "sec.gov", "edgar.sec.gov", "investor.gov", "federalreserve.gov",
            "bls.gov", "census.gov", "treasury.gov", "nasdaq.com", "nyse.com"
        ]
        self.session = None
    
    async def initialize(self):
        """Initialize the fact-checking service"""
        self.session = aiohttp.ClientSession()
        await super().initialize()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        await super().cleanup()
    
    def extract_numerical_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical claims from text"""
        claims = []
        
        # Patterns for financial metrics
        patterns = {
            "revenue": r"revenue.*?\$([\d,\.]+)\s*(billion|million|thousand)?",
            "profit": r"profit.*?\$([\d,\.]+)\s*(billion|million|thousand)?",
            "earnings": r"earnings.*?\$([\d,\.]+)\s*(billion|million|thousand)?",
            "pe_ratio": r"p/e.*?ratio.*?([\d\.]+)",
            "market_cap": r"market.*?cap.*?\$([\d,\.]+)\s*(billion|million|thousand)?",
            "growth": r"growth.*?([\d\.]+)%",
            "margin": r"margin.*?([\d\.]+)%",
            "debt": r"debt.*?\$([\d,\.]+)\s*(billion|million|thousand)?"
        }
        
        text_lower = text.lower()
        
        for metric, pattern in patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                claims.append({
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "context": match.group(0),
                    "position": match.span()
                })
        
        return claims
    
    def extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        claims = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Keywords that often indicate factual claims
        fact_indicators = [
            "reported", "announced", "disclosed", "filed", "stated", "confirmed",
            "according to", "data shows", "results indicate", "study found",
            "increased by", "decreased by", "grew by", "fell by", "rose by"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains fact indicators
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in fact_indicators):
                claims.append(sentence)
            
            # Check if sentence contains numerical data
            if re.search(r'\d+', sentence):
                claims.append(sentence)
        
        return claims[:10]  # Limit to top 10 claims
    
    async def verify_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Verify factual claims against reliable sources"""
        try:
            verification_results = []
            
            for claim in claims:
                # Extract numerical claims
                numerical_claims = self.extract_numerical_claims(claim)
                
                # Simplified verification (in real implementation, would cross-reference with databases)
                verification = {
                    "claim": claim,
                    "verified": None,  # True/False/None (unknown)
                    "confidence": 0.5,  # 0-1 scale
                    "sources_checked": [],
                    "contradictions": [],
                    "supporting_evidence": [],
                    "numerical_claims": numerical_claims
                }
                
                # Basic verification logic
                if numerical_claims:
                    # Check if numerical values are reasonable
                    for num_claim in numerical_claims:
                        try:
                            value = float(num_claim["value"].replace(",", ""))
                            metric = num_claim["metric"]
                            
                            # Basic sanity checks
                            if metric == "pe_ratio" and (value < 0 or value > 1000):
                                verification["contradictions"].append(f"Unrealistic P/E ratio: {value}")
                            elif metric == "growth" and abs(value) > 1000:
                                verification["contradictions"].append(f"Unrealistic growth rate: {value}%")
                            elif metric == "margin" and (value < -100 or value > 100):
                                verification["contradictions"].append(f"Unrealistic margin: {value}%")
                            else:
                                verification["supporting_evidence"].append(f"Reasonable {metric} value: {value}")
                        except ValueError:
                            verification["contradictions"].append(f"Invalid numerical value: {num_claim['value']}")
                
                # Determine verification status
                if verification["contradictions"]:
                    verification["verified"] = False
                    verification["confidence"] = 0.2
                elif verification["supporting_evidence"]:
                    verification["verified"] = True
                    verification["confidence"] = 0.7
                else:
                    verification["verified"] = None
                    verification["confidence"] = 0.5
                
                verification_results.append(verification)
            
            # Calculate overall statistics
            verified_count = sum(1 for v in verification_results if v["verified"] is True)
            disputed_count = sum(1 for v in verification_results if v["verified"] is False)
            unknown_count = sum(1 for v in verification_results if v["verified"] is None)
            
            return {
                "total_claims": len(claims),
                "verified_claims": verified_count,
                "disputed_claims": disputed_count,
                "unknown_claims": unknown_count,
                "verification_results": verification_results,
                "overall_reliability": verified_count / len(claims) if claims else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in claim verification: {e}")
            return {"error": str(e)}
    
    async def cross_reference_historical(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference metrics with historical data"""
        try:
            # Simplified historical cross-reference
            # In real implementation, would query historical databases
            
            cross_ref_results = {
                "metrics_checked": list(metrics.keys()),
                "historical_consistency": {},
                "anomalies_detected": [],
                "confidence_score": 0.7
            }
            
            # Check each metric against historical norms
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Define reasonable ranges for common metrics
                    ranges = {
                        "pe_ratio": (5, 50),
                        "profit_margin": (-0.5, 0.5),
                        "debt_to_equity": (0, 5),
                        "current_ratio": (0.5, 10),
                        "roe": (-0.5, 1.0)
                    }
                    
                    if metric in ranges:
                        min_val, max_val = ranges[metric]
                        if min_val <= value <= max_val:
                            cross_ref_results["historical_consistency"][metric] = "normal"
                        else:
                            cross_ref_results["historical_consistency"][metric] = "anomalous"
                            cross_ref_results["anomalies_detected"].append(
                                f"{metric} value {value} outside normal range [{min_val}, {max_val}]"
                            )
            
            return cross_ref_results
            
        except Exception as e:
            logger.error(f"Error in historical cross-reference: {e}")
            return {"error": str(e)}
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fact-checking validation"""
        try:
            # Extract text content for fact-checking
            text_content = ""
            
            # Collect text from various sources
            if "analysis_results" in data:
                text_content += str(data["analysis_results"])
            
            if "research_data" in data:
                text_content += str(data["research_data"])
            
            if not text_content:
                return {"error": "No text content available for fact-checking"}
            
            # Extract claims
            factual_claims = self.extract_factual_claims(text_content)
            
            if not factual_claims:
                return {
                    "message": "No factual claims detected for verification",
                    "confidence": 0.8
                }
            
            # Verify claims
            verification_results = await self.verify_claims(factual_claims)
            
            return {
                "validation_type": "fact_checking",
                "claims_extracted": len(factual_claims),
                "verification_results": verification_results,
                "overall_reliability": verification_results.get("overall_reliability", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in fact-checking validation: {e}")
            return {"error": str(e)}

class BiasDetectionService(BaseValidationService):
    """Service for detecting cognitive biases in analysis"""
    
    def __init__(self):
        super().__init__("BiasDetection")
        self.bias_patterns = {
            "confirmation_bias": [
                r"only.*positive", r"ignore.*negative", r"cherry.*pick",
                r"selective.*evidence", r"confirms.*belief"
            ],
            "anchoring_bias": [
                r"first.*impression", r"initial.*estimate", r"starting.*point",
                r"reference.*price", r"historical.*high"
            ],
            "availability_bias": [
                r"recent.*events", r"memorable.*case", r"vivid.*example",
                r"comes.*to.*mind", r"easy.*to.*recall"
            ],
            "overconfidence_bias": [
                r"certain.*that", r"definitely.*will", r"guaranteed.*to",
                r"no.*doubt", r"absolutely.*sure"
            ],
            "herding_bias": [
                r"everyone.*thinks", r"market.*consensus", r"popular.*opinion",
                r"crowd.*believes", r"majority.*agrees"
            ],
            "loss_aversion": [
                r"avoid.*loss", r"fear.*of.*losing", r"protect.*capital",
                r"minimize.*downside", r"risk.*averse"
            ]
        }
    
    def detect_emotional_language(self, text: str) -> Dict[str, Any]:
        """Detect emotional language that might indicate bias"""
        emotional_words = {
            "positive": ["amazing", "fantastic", "incredible", "outstanding", "phenomenal", "spectacular"],
            "negative": ["terrible", "awful", "horrible", "disastrous", "catastrophic", "devastating"],
            "fear": ["scary", "frightening", "terrifying", "alarming", "worrying", "concerning"],
            "greed": ["opportunity", "goldmine", "jackpot", "bonanza", "windfall", "fortune"]
        }
        
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, words in emotional_words.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                detected_emotions[emotion] = count
        
        return detected_emotions
    
    def analyze_language_certainty(self, text: str) -> Dict[str, Any]:
        """Analyze the certainty level in language"""
        certainty_indicators = {
            "high_certainty": ["will", "definitely", "certainly", "absolutely", "guaranteed", "sure"],
            "medium_certainty": ["likely", "probably", "expected", "should", "anticipated"],
            "low_certainty": ["might", "could", "possibly", "perhaps", "maybe", "uncertain"]
        }
        
        text_lower = text.lower()
        certainty_analysis = {}
        
        for level, indicators in certainty_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            certainty_analysis[level] = count
        
        # Calculate overall certainty score
        total_indicators = sum(certainty_analysis.values())
        if total_indicators > 0:
            certainty_score = (
                certainty_analysis["high_certainty"] * 1.0 +
                certainty_analysis["medium_certainty"] * 0.5 +
                certainty_analysis["low_certainty"] * 0.0
            ) / total_indicators
        else:
            certainty_score = 0.5  # Neutral
        
        return {
            "certainty_breakdown": certainty_analysis,
            "certainty_score": certainty_score,
            "total_indicators": total_indicators
        }
    
    def detect_bias_patterns(self, text: str) -> Dict[str, Any]:
        """Detect specific bias patterns in text"""
        detected_biases = {}
        text_lower = text.lower()
        
        for bias_type, patterns in self.bias_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            
            if matches:
                detected_biases[bias_type] = {
                    "detected": True,
                    "patterns_matched": matches,
                    "severity": len(matches)
                }
        
        return detected_biases
    
    async def detect_biases(self, text: str) -> Dict[str, Any]:
        """Comprehensive bias detection in text"""
        try:
            # Detect bias patterns
            bias_patterns = self.detect_bias_patterns(text)
            
            # Analyze emotional language
            emotional_analysis = self.detect_emotional_language(text)
            
            # Analyze certainty levels
            certainty_analysis = self.analyze_language_certainty(text)
            
            # Calculate overall bias score
            bias_score = 0
            bias_factors = []
            
            # Score based on detected biases
            for bias_type, details in bias_patterns.items():
                bias_score += details["severity"] * 0.2
                bias_factors.append(f"{bias_type}: {details['severity']} indicators")
            
            # Score based on emotional language
            total_emotional_words = sum(emotional_analysis.values())
            if total_emotional_words > 5:
                bias_score += 0.3
                bias_factors.append(f"High emotional language: {total_emotional_words} words")
            
            # Score based on overconfidence
            if certainty_analysis["certainty_score"] > 0.8:
                bias_score += 0.2
                bias_factors.append(f"High certainty language: {certainty_analysis['certainty_score']:.2f}")
            
            # Determine bias level
            if bias_score > 1.0:
                bias_level = "high"
            elif bias_score > 0.5:
                bias_level = "medium"
            else:
                bias_level = "low"
            
            return {
                "bias_level": bias_level,
                "bias_score": min(bias_score, 2.0),  # Cap at 2.0
                "detected_biases": bias_patterns,
                "emotional_analysis": emotional_analysis,
                "certainty_analysis": certainty_analysis,
                "bias_factors": bias_factors,
                "recommendations": self._generate_bias_recommendations(bias_patterns, bias_level),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in bias detection: {e}")
            return {"error": str(e)}
    
    def _generate_bias_recommendations(self, detected_biases: Dict[str, Any], bias_level: str) -> List[str]:
        """Generate recommendations to mitigate detected biases"""
        recommendations = []
        
        if "confirmation_bias" in detected_biases:
            recommendations.append("Consider contradictory evidence and alternative viewpoints")
        
        if "anchoring_bias" in detected_biases:
            recommendations.append("Re-evaluate initial assumptions and consider multiple reference points")
        
        if "overconfidence_bias" in detected_biases:
            recommendations.append("Acknowledge uncertainty and consider probability ranges")
        
        if "herding_bias" in detected_biases:
            recommendations.append("Develop independent analysis before considering market consensus")
        
        if bias_level == "high":
            recommendations.append("Seek peer review and external validation of analysis")
            recommendations.append("Implement systematic decision-making frameworks")
        
        if not recommendations:
            recommendations.append("Continue maintaining objective analytical approach")
        
        return recommendations
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bias detection validation"""
        try:
            # Extract text content for bias analysis
            text_content = ""
            
            if "analysis_results" in data:
                text_content += str(data["analysis_results"])
            
            if "strategy_results" in data:
                text_content += str(data["strategy_results"])
            
            if not text_content:
                return {"error": "No text content available for bias detection"}
            
            # Perform bias detection
            bias_results = await self.detect_biases(text_content)
            
            return {
                "validation_type": "bias_detection",
                "text_analyzed_length": len(text_content),
                "bias_analysis": bias_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in bias detection validation: {e}")
            return {"error": str(e)}

class DataConsistencyService(BaseValidationService):
    """Service for checking data consistency across sources"""
    
    def __init__(self):
        super().__init__("DataConsistency")
    
    def compare_numerical_values(self, value1: float, value2: float, tolerance: float = 0.05) -> bool:
        """Compare two numerical values with tolerance"""
        if value1 == 0 and value2 == 0:
            return True
        if value1 == 0 or value2 == 0:
            return abs(value1 - value2) < tolerance
        
        relative_diff = abs(value1 - value2) / max(abs(value1), abs(value2))
        return relative_diff <= tolerance
    
    def extract_common_metrics(self, data_sources: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract common metrics from different data sources"""
        common_metrics = {}
        
        for source_name, source_data in data_sources.items():
            metrics = {}
            
            # Extract metrics from different data structures
            if isinstance(source_data, dict):
                # Look for fundamental metrics
                fundamentals = source_data.get("fundamentals", {})
                if isinstance(fundamentals, dict):
                    for key, value in fundamentals.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            metrics[key] = float(value)
                
                # Look for price data
                if "current_price" in source_data:
                    metrics["current_price"] = float(source_data["current_price"])
                
                # Look for other numerical data
                for key, value in source_data.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[key] = float(value)
            
            common_metrics[source_name] = metrics
        
        return common_metrics
    
    def find_inconsistencies(self, metrics_by_source: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Find inconsistencies between data sources"""
        inconsistencies = []
        
        # Get all unique metric names
        all_metrics = set()
        for metrics in metrics_by_source.values():
            all_metrics.update(metrics.keys())
        
        # Check each metric across sources
        for metric in all_metrics:
            sources_with_metric = {}
            for source, metrics in metrics_by_source.items():
                if metric in metrics:
                    sources_with_metric[source] = metrics[metric]
            
            if len(sources_with_metric) < 2:
                continue  # Need at least 2 sources to compare
            
            # Compare all pairs of sources
            source_names = list(sources_with_metric.keys())
            for i in range(len(source_names)):
                for j in range(i + 1, len(source_names)):
                    source1, source2 = source_names[i], source_names[j]
                    value1, value2 = sources_with_metric[source1], sources_with_metric[source2]
                    
                    if not self.compare_numerical_values(value1, value2):
                        inconsistencies.append({
                            "metric": metric,
                            "source1": source1,
                            "value1": value1,
                            "source2": source2,
                            "value2": value2,
                            "difference": abs(value1 - value2),
                            "relative_difference": abs(value1 - value2) / max(abs(value1), abs(value2)) if max(abs(value1), abs(value2)) > 0 else 0
                        })
        
        return inconsistencies
    
    async def check_consistency(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency across multiple data sources"""
        try:
            # Extract common metrics
            metrics_by_source = self.extract_common_metrics(data_sources)
            
            if len(metrics_by_source) < 2:
                return {
                    "message": "Need at least 2 data sources for consistency checking",
                    "sources_available": len(metrics_by_source)
                }
            
            # Find inconsistencies
            inconsistencies = self.find_inconsistencies(metrics_by_source)
            
            # Calculate consistency score
            total_comparisons = 0
            consistent_comparisons = 0
            
            all_metrics = set()
            for metrics in metrics_by_source.values():
                all_metrics.update(metrics.keys())
            
            for metric in all_metrics:
                sources_with_metric = [s for s, m in metrics_by_source.items() if metric in m]
                if len(sources_with_metric) >= 2:
                    # Count all possible pairs
                    pairs = len(sources_with_metric) * (len(sources_with_metric) - 1) // 2
                    total_comparisons += pairs
                    
                    # Count consistent pairs
                    inconsistent_pairs = len([inc for inc in inconsistencies if inc["metric"] == metric])
                    consistent_comparisons += pairs - inconsistent_pairs
            
            consistency_score = consistent_comparisons / total_comparisons if total_comparisons > 0 else 1.0
            
            return {
                "consistency_score": consistency_score,
                "total_comparisons": total_comparisons,
                "consistent_comparisons": consistent_comparisons,
                "inconsistencies_found": len(inconsistencies),
                "inconsistencies": inconsistencies,
                "sources_analyzed": list(metrics_by_source.keys()),
                "metrics_compared": list(all_metrics),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in consistency checking: {e}")
            return {"error": str(e)}
    
    async def validate_statistics(self, statistical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical claims and methodologies"""
        try:
            validation_results = {
                "statistical_validity": True,
                "issues_found": [],
                "recommendations": []
            }
            
            # Check for common statistical issues
            for key, value in statistical_data.items():
                if isinstance(value, (int, float)):
                    # Check for impossible values
                    if key.endswith("_ratio") and value < 0:
                        validation_results["issues_found"].append(f"Negative ratio value: {key} = {value}")
                        validation_results["statistical_validity"] = False
                    
                    if key.endswith("_percentage") and (value < 0 or value > 100):
                        validation_results["issues_found"].append(f"Invalid percentage: {key} = {value}")
                        validation_results["statistical_validity"] = False
                    
                    # Check for extreme values
                    if abs(value) > 1e10:
                        validation_results["issues_found"].append(f"Extremely large value: {key} = {value}")
                    
                    # Check for NaN or infinite values
                    if np.isnan(value) or np.isinf(value):
                        validation_results["issues_found"].append(f"Invalid numerical value: {key} = {value}")
                        validation_results["statistical_validity"] = False
            
            # Generate recommendations
            if validation_results["issues_found"]:
                validation_results["recommendations"].append("Review data collection and calculation methods")
                validation_results["recommendations"].append("Verify data sources and ensure proper data cleaning")
            else:
                validation_results["recommendations"].append("Statistical data appears valid")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in statistical validation: {e}")
            return {"error": str(e)}
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data consistency validation"""
        try:
            # Check consistency across data sources
            consistency_results = await self.check_consistency(data)
            
            # Validate statistical data if present
            statistical_validation = await self.validate_statistics(data)
            
            return {
                "validation_type": "data_consistency",
                "consistency_analysis": consistency_results,
                "statistical_validation": statistical_validation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in data consistency validation: {e}")
            return {"error": str(e)}

class SourceReliabilityService(BaseValidationService):
    """Service for assessing the reliability of information sources"""
    
    def __init__(self):
        super().__init__("SourceReliability")
        self.reliable_domains = {
            "sec.gov": 0.95,
            "edgar.sec.gov": 0.95,
            "investor.gov": 0.9,
            "federalreserve.gov": 0.95,
            "nasdaq.com": 0.85,
            "nyse.com": 0.85,
            "bloomberg.com": 0.8,
            "reuters.com": 0.8,
            "wsj.com": 0.75,
            "ft.com": 0.75,
            "cnbc.com": 0.7,
            "marketwatch.com": 0.7,
            "yahoo.com": 0.6,
            "finance.yahoo.com": 0.65
        }
    
    def assess_source_credibility(self, source: str) -> Dict[str, Any]:
        """Assess the credibility of a single source"""
        credibility_score = 0.5  # Default neutral score
        credibility_factors = []
        
        source_lower = source.lower()
        
        # Check against known reliable domains
        for domain, score in self.reliable_domains.items():
            if domain in source_lower:
                credibility_score = score
                credibility_factors.append(f"Known reliable domain: {domain}")
                break
        
        # Check for government sources
        if ".gov" in source_lower:
            credibility_score = max(credibility_score, 0.9)
            credibility_factors.append("Government source")
        
        # Check for academic sources
        if ".edu" in source_lower:
            credibility_score = max(credibility_score, 0.8)
            credibility_factors.append("Academic source")
        
        # Check for potential red flags
        red_flags = ["blog", "forum", "reddit", "twitter", "facebook", "unknown"]
        for flag in red_flags:
            if flag in source_lower:
                credibility_score = min(credibility_score, 0.4)
                credibility_factors.append(f"Potential reliability concern: {flag}")
        
        # Determine credibility level
        if credibility_score >= 0.8:
            credibility_level = "high"
        elif credibility_score >= 0.6:
            credibility_level = "medium"
        else:
            credibility_level = "low"
        
        return {
            "source": source,
            "credibility_score": credibility_score,
            "credibility_level": credibility_level,
            "credibility_factors": credibility_factors
        }
    
    async def assess_sources(self, sources: List[str]) -> Dict[str, Any]:
        """Assess the reliability of multiple sources"""
        try:
            source_assessments = []
            total_score = 0
            
            for source in sources:
                assessment = self.assess_source_credibility(source)
                source_assessments.append(assessment)
                total_score += assessment["credibility_score"]
            
            # Calculate overall reliability
            overall_reliability = total_score / len(sources) if sources else 0
            
            # Count sources by credibility level
            credibility_distribution = {
                "high": len([a for a in source_assessments if a["credibility_level"] == "high"]),
                "medium": len([a for a in source_assessments if a["credibility_level"] == "medium"]),
                "low": len([a for a in source_assessments if a["credibility_level"] == "low"])
            }
            
            # Generate recommendations
            recommendations = []
            if credibility_distribution["low"] > len(sources) * 0.3:
                recommendations.append("Consider seeking additional high-credibility sources")
            if credibility_distribution["high"] == 0:
                recommendations.append("Include at least one high-credibility source (government, academic, or major financial institution)")
            if overall_reliability < 0.6:
                recommendations.append("Overall source reliability is below recommended threshold")
            
            if not recommendations:
                recommendations.append("Source reliability appears adequate")
            
            return {
                "total_sources": len(sources),
                "overall_reliability": overall_reliability,
                "credibility_distribution": credibility_distribution,
                "source_assessments": source_assessments,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in source reliability assessment: {e}")
            return {"error": str(e)}
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform source reliability validation"""
        try:
            # Extract sources from data
            sources = []
            
            # Look for sources in different parts of the data
            if "sources_used" in data:
                sources.extend(data["sources_used"])
            
            if "research_data" in data:
                research_data = data["research_data"]
                if isinstance(research_data, dict):
                    for key, value in research_data.items():
                        if isinstance(value, dict) and "source" in value:
                            sources.append(value["source"])
            
            # Look for URLs or domain names in the data
            data_str = str(data)
            url_pattern = r'https?://([\w\.-]+)'
            urls = re.findall(url_pattern, data_str)
            sources.extend(urls)
            
            if not sources:
                return {
                    "message": "No sources found for reliability assessment",
                    "validation_type": "source_reliability"
                }
            
            # Remove duplicates
            sources = list(set(sources))
            
            # Assess source reliability
            reliability_results = await self.assess_sources(sources)
            
            return {
                "validation_type": "source_reliability",
                "sources_found": len(sources),
                "reliability_assessment": reliability_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in source reliability validation: {e}")
            return {"error": str(e)}