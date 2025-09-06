from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ...models.requests import ValidationRequest
from ...models.responses import ValidationResponse
from ...core.database import get_db, get_redis

router = APIRouter()

@router.post("/execute", response_model=ValidationResponse)
async def execute_validation(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    redis_client=Depends(get_redis)
):
    """Execute validation task using the Validation Agent"""
    try:
        logger.info(f"Executing validation request for content: {request.content[:100]}...")
        
        # Mock validation result
        result = {
            "fact_check_results": {
                "verified_facts": 8,
                "disputed_facts": 1,
                "unverifiable_claims": 2,
                "overall_accuracy": 0.85,
                "fact_details": [
                    {
                        "claim": "Company revenue increased by 15%",
                        "status": "verified",
                        "confidence": 0.95,
                        "sources": ["SEC filing 10-K", "earnings report"]
                    },
                    {
                        "claim": "Market share is 25%",
                        "status": "disputed",
                        "confidence": 0.60,
                        "sources": ["industry report"],
                        "alternative_data": "Market research suggests 22%"
                    }
                ]
            },
            "bias_detection": {
                "overall_bias_score": 0.3,  # 0 = no bias, 1 = high bias
                "bias_types": {
                    "confirmation_bias": 0.2,
                    "selection_bias": 0.4,
                    "anchoring_bias": 0.1,
                    "availability_bias": 0.3
                },
                "bias_indicators": [
                    "Selective use of positive metrics",
                    "Limited time frame analysis"
                ],
                "recommendations": [
                    "Include longer historical data",
                    "Consider negative indicators",
                    "Add peer comparison"
                ]
            },
            "source_credibility": {
                "high_credibility": 6,
                "medium_credibility": 3,
                "low_credibility": 1,
                "source_breakdown": {
                    "SEC filings": "high",
                    "earnings reports": "high",
                    "analyst reports": "medium",
                    "social media": "low"
                }
            },
            "confidence_score": 0.82,
            "validation_summary": "Content shows good factual accuracy with minor bias concerns"
        }
        
        return ValidationResponse(
            request_id=request.request_id,
            content_hash=hash(request.content),
            validation_type=request.validation_type,
            results=result,
            status="completed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error executing validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fact-check")
async def fact_check_claims(
    claims: List[str],
    context: Optional[str] = None,
    db=Depends(get_db)
):
    """Fact-check specific claims"""
    try:
        results = []
        for i, claim in enumerate(claims):
            # Mock fact-checking
            result = {
                "claim": claim,
                "status": "verified" if i % 3 != 0 else "disputed",
                "confidence": 0.9 if i % 3 != 0 else 0.4,
                "sources": [f"source_{i+1}", f"source_{i+2}"],
                "explanation": f"Analysis of claim: {claim[:50]}..."
            }
            results.append(result)
        
        return {
            "claims_checked": len(claims),
            "verified": len([r for r in results if r["status"] == "verified"]),
            "disputed": len([r for r in results if r["status"] == "disputed"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fact-checking claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bias-analysis")
async def analyze_bias(
    content: str,
    analysis_type: str = "comprehensive",
    db=Depends(get_db)
):
    """Analyze content for various types of bias"""
    try:
        # Mock bias analysis
        bias_analysis = {
            "content_length": len(content),
            "analysis_type": analysis_type,
            "bias_score": 0.35,
            "bias_breakdown": {
                "confirmation_bias": {
                    "score": 0.4,
                    "indicators": ["Cherry-picked data points", "Ignored contradictory evidence"]
                },
                "anchoring_bias": {
                    "score": 0.2,
                    "indicators": ["Heavy reliance on initial data point"]
                },
                "availability_bias": {
                    "score": 0.3,
                    "indicators": ["Recent events overweighted"]
                }
            },
            "recommendations": [
                "Include more diverse data sources",
                "Consider longer time horizons",
                "Add contrarian viewpoints"
            ],
            "confidence": 0.78,
            "timestamp": datetime.now().isoformat()
        }
        
        return bias_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))