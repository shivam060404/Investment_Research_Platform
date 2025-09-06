from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger
import asyncio
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

class BaseStrategyService(ABC):
    """Base class for all strategy services"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the strategy service"""
        self.is_initialized = True
        logger.info(f"{self.name} strategy service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} strategy service cleaned up")
    
    @abstractmethod
    async def generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy based on the provided data"""
        pass

class PortfolioOptimizationService(BaseStrategyService):
    """Service for portfolio optimization and asset allocation"""
    
    def __init__(self):
        super().__init__("PortfolioOptimization")
    
    def calculate_expected_returns(self, historical_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate expected returns for assets"""
        expected_returns = {}
        
        for asset, prices in historical_data.items():
            if len(prices) < 2:
                expected_returns[asset] = 0.0
                continue
            
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            expected_returns[asset] = np.mean(returns) * 252
        
        return expected_returns
    
    def calculate_covariance_matrix(self, historical_data: Dict[str, List[float]]) -> np.ndarray:
        """Calculate covariance matrix for assets"""

        returns_data = []
        asset_names = list(historical_data.keys())
        
        for asset in asset_names:
            prices = historical_data[asset]
            if len(prices) < 2:
                returns_data.append([0.0] * max(1, len(prices) - 1))
            else:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                returns_data.append(returns)
        

        min_length = min(len(returns) for returns in returns_data)
        returns_matrix = np.array([returns[:min_length] for returns in returns_data])
        
        if returns_matrix.shape[1] < 2:

            return np.eye(len(asset_names)) * 0.01
        

        try:
            cov_estimator = LedoitWolf()
            covariance_matrix = cov_estimator.fit(returns_matrix.T).covariance_
            return covariance_matrix * 252  # Annualized
        except:

            return np.cov(returns_matrix) * 252
    
    def optimize_portfolio(self, expected_returns: Dict[str, float], covariance_matrix: np.ndarray, 
                          risk_tolerance: float = 0.5, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization"""
        assets = list(expected_returns.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {"error": "No assets provided for optimization"}
        
        # Convert expected returns to array
        returns_array = np.array([expected_returns[asset] for asset in assets])
        
        # Objective function: maximize utility (return - risk_penalty * variance)
        def objective(weights):
            portfolio_return = np.dot(weights, returns_array)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            # Utility function: return - risk_penalty * variance
            risk_penalty = (1 - risk_tolerance) * 10  # Scale risk penalty
            return -(portfolio_return - risk_penalty * portfolio_variance)
        
        # Constraints
        constraints_list = []
        
        # Weights sum to 1
        constraints_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Additional constraints
        if constraints:
            # Maximum weight per asset
            max_weight = constraints.get('max_weight_per_asset', 0.4)
            for i in range(n_assets):
                constraints_list.append({'type': 'ineq', 'fun': lambda x, i=i: max_weight - x[i]})
            
            # Minimum weight per asset (if specified)
            min_weight = constraints.get('min_weight_per_asset', 0.0)
            for i in range(n_assets):
                constraints_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - min_weight})
        
        # Bounds (0 to 1 for long-only portfolio)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints_list)
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(optimal_weights, returns_array)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Create allocation dictionary
                allocation = {assets[i]: float(optimal_weights[i]) for i in range(n_assets)}
                
                return {
                    "allocation": allocation,
                    "expected_return": float(portfolio_return),
                    "expected_volatility": float(portfolio_volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "optimization_success": True,
                    "optimization_message": "Portfolio optimization successful"
                }
            else:
                return {
                    "error": "Portfolio optimization failed",
                    "optimization_message": result.message,
                    "optimization_success": False
                }
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Fallback to equal weights
            equal_weights = {asset: 1.0 / n_assets for asset in assets}
            return {
                "allocation": equal_weights,
                "expected_return": float(np.mean(returns_array)),
                "expected_volatility": 0.15,  # Default assumption
                "sharpe_ratio": 0.5,  # Default assumption
                "optimization_success": False,
                "optimization_message": f"Optimization failed, using equal weights: {str(e)}"
            }
    
    async def optimize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio optimization"""
        try:
            # Extract relevant data
            analysis_data = data.get("analysis_data", {})
            validation_data = data.get("validation_data", {})
            
            # Get symbols to optimize
            symbols = data.get("symbols", [])
            if not symbols:
                # Try to extract from analysis data
                if "research_data_summary" in analysis_data:
                    symbols = analysis_data["research_data_summary"].get("symbols_analyzed", [])
            
            if not symbols:
                return {"error": "No symbols provided for portfolio optimization"}
            
            # Mock historical data (in real implementation, would fetch actual data)
            historical_data = {}
            for symbol in symbols:
                # Generate mock price data
                np.random.seed(hash(symbol) % 2**32)  # Deterministic random data
                base_price = 100
                returns = np.random.normal(0.0008, 0.02, 252)  # Daily returns
                prices = [base_price]
                for ret in returns:
                    prices.append(prices[-1] * (1 + ret))
                historical_data[symbol] = prices
            
            # Calculate expected returns and covariance
            expected_returns = self.calculate_expected_returns(historical_data)
            covariance_matrix = self.calculate_covariance_matrix(historical_data)
            
            # Get risk tolerance
            risk_tolerance = data.get("risk_tolerance", 0.5)
            if isinstance(risk_tolerance, str):
                risk_map = {"conservative": 0.3, "moderate": 0.5, "aggressive": 0.8}
                risk_tolerance = risk_map.get(risk_tolerance.lower(), 0.5)
            
            # Set up constraints
            constraints = {
                "max_weight_per_asset": data.get("max_position_size", 0.3),
                "min_weight_per_asset": 0.05  # Minimum 5% allocation
            }
            
            # Optimize portfolio
            optimization_result = self.optimize_portfolio(
                expected_returns, covariance_matrix, risk_tolerance, constraints
            )
            
            # Add additional analysis
            if optimization_result.get("optimization_success"):
                allocation = optimization_result["allocation"]
                
                # Calculate diversification metrics
                weights = np.array(list(allocation.values()))
                herfindahl_index = np.sum(weights ** 2)
                effective_assets = 1 / herfindahl_index
                
                optimization_result.update({
                    "diversification_metrics": {
                        "herfindahl_index": float(herfindahl_index),
                        "effective_number_of_assets": float(effective_assets),
                        "concentration_risk": "High" if herfindahl_index > 0.5 else "Medium" if herfindahl_index > 0.25 else "Low"
                    },
                    "rebalancing_frequency": "Monthly",
                    "implementation_notes": [
                        "Consider transaction costs when rebalancing",
                        "Monitor correlation changes over time",
                        "Review allocation quarterly"
                    ]
                })
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {"error": str(e)}
    
    async def generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio optimization strategy"""
        return await self.optimize(data)

class RiskManagementService(BaseStrategyService):
    """Service for risk management and assessment"""
    
    def __init__(self):
        super().__init__("RiskManagement")
    
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if not returns:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        tail_returns = [r for r in returns if r <= var]
        return np.mean(tail_returns) if tail_returns else var
    
    def assess_portfolio_risk(self, allocation: Dict[str, float], risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        risk_assessment = {
            "overall_risk_level": "medium",
            "risk_factors": [],
            "risk_score": 0.5,
            "recommendations": []
        }
        
        # Analyze concentration risk
        max_allocation = max(allocation.values()) if allocation else 0
        if max_allocation > 0.4:
            risk_assessment["risk_factors"].append(f"High concentration risk: {max_allocation:.1%} in single asset")
            risk_assessment["risk_score"] += 0.2
        
        # Analyze volatility risk
        portfolio_volatility = risk_metrics.get("expected_volatility", 0.15)
        if portfolio_volatility > 0.25:
            risk_assessment["risk_factors"].append(f"High volatility: {portfolio_volatility:.1%} annualized")
            risk_assessment["risk_score"] += 0.2
        elif portfolio_volatility > 0.20:
            risk_assessment["risk_factors"].append(f"Moderate volatility: {portfolio_volatility:.1%} annualized")
            risk_assessment["risk_score"] += 0.1
        
        # Analyze Sharpe ratio
        sharpe_ratio = risk_metrics.get("sharpe_ratio", 0.5)
        if sharpe_ratio < 0.5:
            risk_assessment["risk_factors"].append(f"Low risk-adjusted returns: Sharpe ratio {sharpe_ratio:.2f}")
            risk_assessment["risk_score"] += 0.1
        
        # Determine overall risk level
        if risk_assessment["risk_score"] > 0.7:
            risk_assessment["overall_risk_level"] = "high"
        elif risk_assessment["risk_score"] < 0.3:
            risk_assessment["overall_risk_level"] = "low"
        
        # Generate recommendations
        if max_allocation > 0.3:
            risk_assessment["recommendations"].append("Consider diversifying to reduce concentration risk")
        if portfolio_volatility > 0.25:
            risk_assessment["recommendations"].append("Consider adding low-volatility assets to reduce portfolio risk")
        if sharpe_ratio < 0.5:
            risk_assessment["recommendations"].append("Review asset selection to improve risk-adjusted returns")
        
        if not risk_assessment["recommendations"]:
            risk_assessment["recommendations"].append("Risk profile appears balanced")
        
        return risk_assessment
    
    def generate_hedging_strategies(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hedging strategies for portfolio protection"""
        hedging_strategies = []
        
        # Market hedge
        hedging_strategies.append({
            "strategy_type": "market_hedge",
            "description": "Use index ETFs or futures to hedge market risk",
            "instruments": ["SPY puts", "VIX calls", "Market index futures"],
            "hedge_ratio": 0.3,  # Hedge 30% of market exposure
            "cost_estimate": "1-2% of portfolio value annually",
            "effectiveness": "High for market downturns"
        })
        
        # Volatility hedge
        hedging_strategies.append({
            "strategy_type": "volatility_hedge",
            "description": "Protect against volatility spikes",
            "instruments": ["VIX options", "Volatility ETFs"],
            "hedge_ratio": 0.1,  # Small allocation to vol protection
            "cost_estimate": "0.5-1% of portfolio value annually",
            "effectiveness": "High during volatility spikes"
        })
        
        # Currency hedge (if international exposure)
        hedging_strategies.append({
            "strategy_type": "currency_hedge",
            "description": "Hedge foreign exchange risk",
            "instruments": ["Currency forwards", "Currency ETFs"],
            "hedge_ratio": 0.8,  # Hedge 80% of FX exposure
            "cost_estimate": "0.2-0.5% of international allocation annually",
            "effectiveness": "High for currency movements"
        })
        
        # Sector rotation hedge
        hedging_strategies.append({
            "strategy_type": "sector_hedge",
            "description": "Hedge sector-specific risks",
            "instruments": ["Sector ETFs", "Pairs trading"],
            "hedge_ratio": 0.2,  # Hedge 20% of sector exposure
            "cost_estimate": "0.5-1% of portfolio value annually",
            "effectiveness": "Medium for sector-specific risks"
        })
        
        return hedging_strategies
    
    async def assess_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            # Extract portfolio allocation
            allocation = data.get("allocation", {})
            if not allocation:
                # Try to extract from strategy data
                strategy_data = data.get("strategy_data", {})
                allocation = strategy_data.get("allocation", {})
            
            if not allocation:
                return {"error": "No portfolio allocation data available for risk assessment"}
            
            # Extract risk metrics
            risk_metrics = data.get("risk_metrics", {})
            if not risk_metrics:
                # Use default metrics
                risk_metrics = {
                    "expected_volatility": 0.15,
                    "sharpe_ratio": 0.6,
                    "max_drawdown": 0.12
                }
            
            # Assess portfolio risk
            portfolio_risk = self.assess_portfolio_risk(allocation, risk_metrics)
            
            # Generate hedging strategies
            hedging_strategies = self.generate_hedging_strategies({"allocation": allocation})
            
            # Calculate position sizing recommendations
            portfolio_value = data.get("portfolio_value", 100000)
            risk_tolerance = data.get("risk_tolerance", 0.02)  # 2% risk per position
            
            position_sizing = {}
            for asset, weight in allocation.items():
                position_value = portfolio_value * weight
                # Simple position sizing based on volatility
                asset_volatility = 0.20  # Default assumption
                max_position_size = (risk_tolerance * portfolio_value) / asset_volatility
                recommended_size = min(position_value, max_position_size)
                
                position_sizing[asset] = {
                    "target_allocation": weight,
                    "target_value": position_value,
                    "recommended_value": recommended_size,
                    "risk_adjusted": recommended_size != position_value
                }
            
            # Risk monitoring plan
            monitoring_plan = {
                "daily_metrics": ["Portfolio value", "VaR", "Volatility"],
                "weekly_metrics": ["Correlation changes", "Sector exposure", "Risk attribution"],
                "monthly_metrics": ["Performance attribution", "Risk-adjusted returns", "Benchmark comparison"],
                "alert_thresholds": {
                    "daily_loss": 0.03,  # 3% daily loss
                    "weekly_loss": 0.05,  # 5% weekly loss
                    "volatility_spike": 1.5,  # 50% increase in volatility
                    "correlation_breakdown": 0.3  # 30% change in correlations
                }
            }
            
            return {
                "portfolio_risk_assessment": portfolio_risk,
                "hedging_strategies": hedging_strategies,
                "position_sizing": position_sizing,
                "risk_monitoring_plan": monitoring_plan,
                "risk_budget": {
                    "total_risk_budget": risk_tolerance,
                    "allocated_risk": sum(weight * 0.15 for weight in allocation.values()),  # Simplified
                    "remaining_risk_capacity": max(0, risk_tolerance - 0.1)
                },
                "stress_test_scenarios": [
                    {"name": "Market crash", "market_decline": -0.20, "expected_portfolio_impact": -0.15},
                    {"name": "Interest rate spike", "rate_increase": 0.02, "expected_portfolio_impact": -0.08},
                    {"name": "Volatility spike", "vol_increase": 2.0, "expected_portfolio_impact": -0.12}
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"error": str(e)}
    
    async def calculate_position_size(self, risk_tolerance: float, portfolio_value: float, 
                                    confidence_level: float) -> Dict[str, Any]:
        """Calculate optimal position sizes based on risk parameters"""
        try:
            # Kelly Criterion-based position sizing
            # Simplified implementation
            
            base_position_size = risk_tolerance * portfolio_value
            
            # Adjust based on confidence level
            confidence_multiplier = confidence_level
            adjusted_position_size = base_position_size * confidence_multiplier
            
            # Position sizing rules
            position_sizing_rules = {
                "max_single_position": min(adjusted_position_size, portfolio_value * 0.2),  # Max 20%
                "min_position_size": portfolio_value * 0.01,  # Min 1%
                "recommended_position": adjusted_position_size,
                "position_as_percentage": (adjusted_position_size / portfolio_value) * 100,
                "risk_per_position": risk_tolerance * 100,
                "confidence_adjustment": confidence_multiplier
            }
            
            # Position sizing methodology
            methodology = {
                "approach": "Risk-based position sizing",
                "factors_considered": [
                    "Portfolio risk tolerance",
                    "Analysis confidence level",
                    "Maximum position limits",
                    "Diversification requirements"
                ],
                "formula": "Position Size = Risk Tolerance × Portfolio Value × Confidence Level",
                "limitations": [
                    "Does not account for correlation with existing positions",
                    "Assumes normal distribution of returns",
                    "Static risk tolerance assumption"
                ]
            }
            
            return {
                "position_sizing_rules": position_sizing_rules,
                "methodology": methodology,
                "recommendations": [
                    f"Start with {position_sizing_rules['position_as_percentage']:.1f}% allocation",
                    "Monitor position correlation with existing holdings",
                    "Adjust size based on realized volatility",
                    "Consider scaling in over time to reduce timing risk"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in position size calculation: {e}")
            return {"error": str(e)}
    
    async def generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management strategy"""
        return await self.assess_risks(data)

class AssetAllocationService(BaseStrategyService):
    """Service for strategic asset allocation"""
    
    def __init__(self):
        super().__init__("AssetAllocation")
    
    def get_strategic_allocation_models(self) -> Dict[str, Dict[str, float]]:
        """Get predefined strategic allocation models"""
        return {
            "conservative": {
                "stocks": 0.30,
                "bonds": 0.50,
                "real_estate": 0.10,
                "commodities": 0.05,
                "cash": 0.05
            },
            "moderate": {
                "stocks": 0.60,
                "bonds": 0.25,
                "real_estate": 0.10,
                "commodities": 0.03,
                "cash": 0.02
            },
            "aggressive": {
                "stocks": 0.80,
                "bonds": 0.10,
                "real_estate": 0.05,
                "commodities": 0.03,
                "cash": 0.02
            },
            "growth": {
                "stocks": 0.85,
                "bonds": 0.05,
                "real_estate": 0.05,
                "commodities": 0.03,
                "cash": 0.02
            }
        }
    
    def customize_allocation(self, base_model: Dict[str, float], 
                           preferences: Dict[str, Any]) -> Dict[str, float]:
        """Customize allocation based on investor preferences"""
        allocation = base_model.copy()
        
        # Adjust based on age (if provided)
        age = preferences.get("age")
        if age:
            # Rule of thumb: bond allocation = age
            target_bond_allocation = min(age / 100, 0.6)  # Cap at 60%
            current_bond_allocation = allocation.get("bonds", 0.25)
            
            adjustment = target_bond_allocation - current_bond_allocation
            allocation["bonds"] = target_bond_allocation
            allocation["stocks"] = max(0.2, allocation["stocks"] - adjustment)  # Min 20% stocks
        
        # Adjust based on investment horizon
        horizon = preferences.get("investment_horizon", "medium_term")
        if horizon == "short_term":
            # Increase cash and bonds for short-term
            allocation["cash"] = min(0.2, allocation["cash"] + 0.1)
            allocation["bonds"] = min(0.6, allocation["bonds"] + 0.1)
            allocation["stocks"] = max(0.2, allocation["stocks"] - 0.2)
        elif horizon == "long_term":
            # Increase stocks for long-term growth
            allocation["stocks"] = min(0.9, allocation["stocks"] + 0.1)
            allocation["bonds"] = max(0.05, allocation["bonds"] - 0.05)
            allocation["cash"] = max(0.01, allocation["cash"] - 0.05)
        
        # Normalize to ensure sum equals 1
        total = sum(allocation.values())
        allocation = {k: v / total for k, v in allocation.items()}
        
        return allocation
    
    def generate_tactical_adjustments(self, strategic_allocation: Dict[str, float], 
                                    market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tactical allocation adjustments based on market conditions"""
        adjustments = {}
        reasoning = []
        
        # Market valuation adjustments
        market_valuation = market_conditions.get("market_valuation", "neutral")
        if market_valuation == "overvalued":
            adjustments["stocks"] = -0.05  # Reduce stocks by 5%
            adjustments["bonds"] = 0.03   # Increase bonds by 3%
            adjustments["cash"] = 0.02    # Increase cash by 2%
            reasoning.append("Reduced equity exposure due to overvaluation")
        elif market_valuation == "undervalued":
            adjustments["stocks"] = 0.05   # Increase stocks by 5%
            adjustments["bonds"] = -0.03  # Reduce bonds by 3%
            adjustments["cash"] = -0.02   # Reduce cash by 2%
            reasoning.append("Increased equity exposure due to undervaluation")
        
        # Interest rate environment
        interest_rate_trend = market_conditions.get("interest_rate_trend", "stable")
        if interest_rate_trend == "rising":
            adjustments["bonds"] = adjustments.get("bonds", 0) - 0.03
            adjustments["real_estate"] = adjustments.get("real_estate", 0) + 0.02
            adjustments["commodities"] = adjustments.get("commodities", 0) + 0.01
            reasoning.append("Reduced bond duration due to rising rates")
        elif interest_rate_trend == "falling":
            adjustments["bonds"] = adjustments.get("bonds", 0) + 0.03
            adjustments["stocks"] = adjustments.get("stocks", 0) + 0.02
            adjustments["cash"] = adjustments.get("cash", 0) - 0.05
            reasoning.append("Increased duration exposure due to falling rates")
        
        # Economic cycle adjustments
        economic_cycle = market_conditions.get("economic_cycle", "expansion")
        if economic_cycle == "recession":
            adjustments["stocks"] = adjustments.get("stocks", 0) - 0.1
            adjustments["bonds"] = adjustments.get("bonds", 0) + 0.05
            adjustments["cash"] = adjustments.get("cash", 0) + 0.05
            reasoning.append("Defensive positioning for recessionary environment")
        elif economic_cycle == "recovery":
            adjustments["stocks"] = adjustments.get("stocks", 0) + 0.08
            adjustments["real_estate"] = adjustments.get("real_estate", 0) + 0.02
            adjustments["bonds"] = adjustments.get("bonds", 0) - 0.05
            adjustments["cash"] = adjustments.get("cash", 0) - 0.05
            reasoning.append("Growth positioning for economic recovery")
        
        # Apply adjustments to strategic allocation
        tactical_allocation = strategic_allocation.copy()
        for asset, adjustment in adjustments.items():
            tactical_allocation[asset] = max(0, tactical_allocation.get(asset, 0) + adjustment)
        
        # Normalize
        total = sum(tactical_allocation.values())
        tactical_allocation = {k: v / total for k, v in tactical_allocation.items()}
        
        return {
            "tactical_allocation": tactical_allocation,
            "adjustments_made": adjustments,
            "reasoning": reasoning,
            "adjustment_magnitude": sum(abs(adj) for adj in adjustments.values())
        }
    
    async def create_allocation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategic asset allocation"""
        try:
            # Determine risk profile
            risk_tolerance = data.get("risk_tolerance", "moderate")
            if isinstance(risk_tolerance, float):
                if risk_tolerance < 0.3:
                    risk_profile = "conservative"
                elif risk_tolerance > 0.7:
                    risk_profile = "aggressive"
                else:
                    risk_profile = "moderate"
            else:
                risk_profile = risk_tolerance.lower()
            
            # Get base allocation model
            allocation_models = self.get_strategic_allocation_models()
            base_allocation = allocation_models.get(risk_profile, allocation_models["moderate"])
            
            # Customize based on preferences
            preferences = {
                "age": data.get("age"),
                "investment_horizon": data.get("investment_horizon", "medium_term"),
                "income_needs": data.get("income_needs", False),
                "tax_considerations": data.get("tax_considerations", {})
            }
            
            strategic_allocation = self.customize_allocation(base_allocation, preferences)
            
            # Generate tactical adjustments
            market_conditions = data.get("market_conditions", {
                "market_valuation": "neutral",
                "interest_rate_trend": "stable",
                "economic_cycle": "expansion"
            })
            
            tactical_results = self.generate_tactical_adjustments(strategic_allocation, market_conditions)
            
            # Implementation guidelines
            implementation = {
                "rebalancing_frequency": "Quarterly",
                "rebalancing_threshold": "5% deviation from target",
                "implementation_approach": "Gradual implementation over 3-6 months",
                "cost_considerations": [
                    "Use low-cost index funds where possible",
                    "Consider tax implications of rebalancing",
                    "Minimize transaction costs"
                ],
                "monitoring_schedule": {
                    "monthly": "Performance review and drift monitoring",
                    "quarterly": "Rebalancing assessment",
                    "annually": "Strategic allocation review"
                }
            }
            
            return {
                "risk_profile": risk_profile,
                "strategic_allocation": strategic_allocation,
                "tactical_allocation": tactical_results["tactical_allocation"],
                "tactical_adjustments": {
                    "adjustments_made": tactical_results["adjustments_made"],
                    "reasoning": tactical_results["reasoning"]
                },
                "implementation_guidelines": implementation,
                "expected_metrics": {
                    "expected_return": self._estimate_expected_return(tactical_results["tactical_allocation"]),
                    "expected_volatility": self._estimate_expected_volatility(tactical_results["tactical_allocation"]),
                    "expected_sharpe_ratio": 0.6  # Placeholder
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in asset allocation: {e}")
            return {"error": str(e)}
    
    def _estimate_expected_return(self, allocation: Dict[str, float]) -> float:
        """Estimate expected return based on allocation"""
        # Historical average returns by asset class
        expected_returns = {
            "stocks": 0.10,
            "bonds": 0.04,
            "real_estate": 0.08,
            "commodities": 0.06,
            "cash": 0.02
        }
        
        return sum(allocation.get(asset, 0) * expected_returns.get(asset, 0.05) 
                  for asset in allocation.keys())
    
    def _estimate_expected_volatility(self, allocation: Dict[str, float]) -> float:
        """Estimate expected volatility based on allocation"""
        # Historical volatilities by asset class
        volatilities = {
            "stocks": 0.16,
            "bonds": 0.04,
            "real_estate": 0.12,
            "commodities": 0.20,
            "cash": 0.01
        }
        
        # Simplified calculation (assumes some correlation)
        weighted_vol = sum(allocation.get(asset, 0) * volatilities.get(asset, 0.10) 
                          for asset in allocation.keys())
        
        # Apply diversification benefit (reduce by 20%)
        return weighted_vol * 0.8
    
    async def generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate asset allocation strategy"""
        return await self.create_allocation(data)

class TradingStrategyService(BaseStrategyService):
    """Service for generating trading strategies"""
    
    def __init__(self):
        super().__init__("TradingStrategy")
    
    def generate_entry_exit_rules(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry and exit rules based on analysis"""
        rules = {
            "entry_rules": [],
            "exit_rules": [],
            "stop_loss_rules": [],
            "take_profit_rules": []
        }
        
        # Technical analysis based rules
        technical_analysis = analysis_data.get("technical_analysis", {})
        if technical_analysis:
            trend = technical_analysis.get("trend", "neutral")
            rsi = technical_analysis.get("indicators", {}).get("rsi")
            
            if trend == "bullish":
                rules["entry_rules"].append("Enter long position on bullish trend confirmation")
                rules["entry_rules"].append("Wait for pullback to support level before entry")
            elif trend == "bearish":
                rules["entry_rules"].append("Consider short position or avoid long entries")
                rules["exit_rules"].append("Exit long positions on bearish trend confirmation")
            
            if rsi:
                if rsi < 30:
                    rules["entry_rules"].append(f"RSI oversold ({rsi:.1f}) - potential buy signal")
                elif rsi > 70:
                    rules["exit_rules"].append(f"RSI overbought ({rsi:.1f}) - consider taking profits")
        
        # Fundamental analysis based rules
        fundamental_analysis = analysis_data.get("fundamental_analysis", {})
        if fundamental_analysis:
            recommendation = fundamental_analysis.get("recommendation", "Hold")
            if recommendation in ["Strong Buy", "Buy"]:
                rules["entry_rules"].append(f"Fundamental analysis supports entry: {recommendation}")
            elif recommendation in ["Strong Sell", "Sell"]:
                rules["exit_rules"].append(f"Fundamental analysis suggests exit: {recommendation}")
        
        # Risk management rules
        rules["stop_loss_rules"] = [
            "Set stop loss at 8% below entry price",
            "Trail stop loss to break-even after 10% gain",
            "Use volatility-based stops (2x ATR)"
        ]
        
        rules["take_profit_rules"] = [
            "Take partial profits at 15% gain",
            "Take additional profits at resistance levels",
            "Let winners run with trailing stops"
        ]
        
        return rules
    
    def generate_position_management_rules(self, risk_tolerance: float) -> Dict[str, Any]:
        """Generate position management rules"""
        return {
            "initial_position_size": f"{min(risk_tolerance * 100, 10):.1f}% of portfolio",
            "scaling_rules": [
                "Start with half position size",
                "Add to position on confirmation signals",
                "Scale out on profit targets"
            ],
            "risk_per_trade": f"{risk_tolerance * 100:.1f}% of portfolio value",
            "maximum_positions": min(10, int(1 / risk_tolerance)),
            "correlation_limits": "Maximum 3 positions in same sector",
            "review_frequency": "Weekly position review and adjustment"
        }
    
    async def generate_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading strategy"""
        try:
            # Extract analysis data
            analysis_data = data.get("analysis_data", {})
            validation_data = data.get("validation_data", {})
            
            # Get confidence level from validation
            confidence_level = validation_data.get("overall_confidence", 0.5)
            
            # Generate entry/exit rules
            trading_rules = self.generate_entry_exit_rules(analysis_data)
            
            # Generate position management rules
            risk_tolerance = data.get("risk_tolerance", 0.02)
            position_management = self.generate_position_management_rules(risk_tolerance)
            
            # Determine strategy type based on analysis
            strategy_type = "balanced"
            if confidence_level > 0.8:
                strategy_type = "aggressive"
            elif confidence_level < 0.4:
                strategy_type = "conservative"
            
            # Time horizon considerations
            investment_horizon = data.get("investment_horizon", "medium_term")
            if investment_horizon == "short_term":
                trading_frequency = "Active (daily to weekly)"
                holding_period = "1-4 weeks"
            elif investment_horizon == "long_term":
                trading_frequency = "Low (monthly to quarterly)"
                holding_period = "6 months to 2 years"
            else:
                trading_frequency = "Moderate (weekly to monthly)"
                holding_period = "1-6 months"
            
            # Strategy implementation
            implementation_plan = {
                "phase_1": "Initial position establishment (Week 1-2)",
                "phase_2": "Position monitoring and adjustment (Ongoing)",
                "phase_3": "Profit taking and rebalancing (As triggered)",
                "review_schedule": {
                    "daily": "Monitor key levels and news",
                    "weekly": "Review positions and performance",
                    "monthly": "Strategy effectiveness review"
                }
            }
            
            # Performance expectations
            expected_performance = {
                "target_return": f"{(confidence_level * 20):.1f}% annually",
                "expected_volatility": f"{(15 + (1-confidence_level) * 10):.1f}% annually",
                "win_rate_target": f"{(50 + confidence_level * 20):.1f}%",
                "risk_reward_ratio": "1:2 minimum",
                "maximum_drawdown": f"{(10 + (1-confidence_level) * 10):.1f}%"
            }
            
            return {
                "strategy_type": strategy_type,
                "confidence_level": confidence_level,
                "trading_rules": trading_rules,
                "position_management": position_management,
                "trading_frequency": trading_frequency,
                "holding_period": holding_period,
                "implementation_plan": implementation_plan,
                "expected_performance": expected_performance,
                "risk_management": {
                    "maximum_portfolio_risk": f"{risk_tolerance * 100:.1f}%",
                    "position_sizing_method": "Risk-based sizing",
                    "stop_loss_strategy": "Volatility-adjusted stops",
                    "diversification_rules": "Maximum 20% in single position"
                },
                "monitoring_requirements": [
                    "Daily price and volume monitoring",
                    "Weekly fundamental update review",
                    "Monthly strategy performance assessment",
                    "Quarterly strategy optimization"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in trading strategy generation: {e}")
            return {"error": str(e)}