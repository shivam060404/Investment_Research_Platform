from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger
import asyncio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from textblob import TextBlob
import re

class BaseAnalysisService(ABC):
    """Base class for all analysis services"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the analysis service"""
        self.is_initialized = True
        logger.info(f"{self.name} analysis service initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} analysis service cleaned up")
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on the provided data"""
        pass

class TechnicalAnalysisService(BaseAnalysisService):
    """Technical analysis service for price and volume data"""
    
    def __init__(self):
        super().__init__("TechnicalAnalysis")
    
    def calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < window:
            return []
        
        sma = []
        for i in range(window - 1, len(prices)):
            avg = sum(prices[i - window + 1:i + 1]) / window
            sma.append(avg)
        return sma
    
    def calculate_ema(self, prices: List[float], window: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < window:
            return []
        
        multiplier = 2 / (window + 1)
        ema = [sum(prices[:window]) / window]  # Start with SMA
        
        for price in prices[window:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
        
        return ema
    
    def calculate_rsi(self, prices: List[float], window: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[:window]) / window
        avg_loss = sum(losses[:window]) / window
        
        rsi = []
        for i in range(window, len(gains)):
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return {"macd": [], "signal": [], "histogram": []}
        
        # Align the EMAs
        min_len = min(len(ema_fast), len(ema_slow))
        ema_fast = ema_fast[-min_len:]
        ema_slow = ema_slow[-min_len:]
        
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(min_len)]
        signal_line = self.calculate_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = []
        if len(signal_line) > 0:
            signal_offset = len(macd_line) - len(signal_line)
            for i in range(len(signal_line)):
                histogram.append(macd_line[signal_offset + i] - signal_line[i])
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def calculate_bollinger_bands(self, prices: List[float], window: int = 20, num_std: float = 2) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < window:
            return {"upper": [], "middle": [], "lower": []}
        
        sma = self.calculate_sma(prices, window)
        
        upper_band = []
        lower_band = []
        
        for i in range(window - 1, len(prices)):
            price_slice = prices[i - window + 1:i + 1]
            std_dev = np.std(price_slice)
            sma_value = sma[i - window + 1]
            
            upper_band.append(sma_value + (num_std * std_dev))
            lower_band.append(sma_value - (num_std * std_dev))
        
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band
        }
    
    def identify_support_resistance(self, prices: List[float], window: int = 5) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        if len(prices) < window * 2 + 1:
            return {"support": [], "resistance": []}
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            # Check for local minima (support)
            is_support = all(prices[i] <= prices[i + j] for j in range(-window, window + 1) if j != 0)
            if is_support:
                support_levels.append(prices[i])
            
            # Check for local maxima (resistance)
            is_resistance = all(prices[i] >= prices[i + j] for j in range(-window, window + 1) if j != 0)
            if is_resistance:
                resistance_levels.append(prices[i])
        
        return {
            "support": list(set(support_levels)),  # Remove duplicates
            "resistance": list(set(resistance_levels))
        }
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            # Extract price data
            historical_data = data.get("historical_data", {})
            if not historical_data:
                return {"error": "No historical data available for technical analysis"}
            
            # Convert to lists if pandas data
            if "Close" in historical_data:
                if isinstance(historical_data["Close"], dict):
                    prices = list(historical_data["Close"].values())
                else:
                    prices = historical_data["Close"].tolist() if hasattr(historical_data["Close"], 'tolist') else list(historical_data["Close"])
            else:
                return {"error": "No closing price data available"}
            
            if len(prices) < 20:
                return {"error": "Insufficient data for technical analysis (minimum 20 data points required)"}
            
            # Calculate technical indicators
            sma_20 = self.calculate_sma(prices, 20)
            sma_50 = self.calculate_sma(prices, 50)
            ema_12 = self.calculate_ema(prices, 12)
            ema_26 = self.calculate_ema(prices, 26)
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            bollinger = self.calculate_bollinger_bands(prices)
            support_resistance = self.identify_support_resistance(prices)
            
            # Current values
            current_price = prices[-1]
            current_rsi = rsi[-1] if rsi else None
            current_macd = macd["macd"][-1] if macd["macd"] else None
            
            # Generate signals
            signals = []
            
            # RSI signals
            if current_rsi:
                if current_rsi > 70:
                    signals.append("RSI indicates overbought condition")
                elif current_rsi < 30:
                    signals.append("RSI indicates oversold condition")
            
            # Moving average signals
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20[-1] > sma_50[-1]:
                    signals.append("Short-term MA above long-term MA (bullish)")
                else:
                    signals.append("Short-term MA below long-term MA (bearish)")
            
            # MACD signals
            if len(macd["histogram"]) > 1:
                if macd["histogram"][-1] > macd["histogram"][-2]:
                    signals.append("MACD histogram increasing (bullish momentum)")
                else:
                    signals.append("MACD histogram decreasing (bearish momentum)")
            
            # Trend analysis
            trend = "neutral"
            if len(prices) >= 10:
                recent_prices = prices[-10:]
                if recent_prices[-1] > recent_prices[0] * 1.02:  # 2% increase
                    trend = "bullish"
                elif recent_prices[-1] < recent_prices[0] * 0.98:  # 2% decrease
                    trend = "bearish"
            
            return {
                "indicators": {
                    "sma_20": sma_20[-1] if sma_20 else None,
                    "sma_50": sma_50[-1] if sma_50 else None,
                    "ema_12": ema_12[-1] if ema_12 else None,
                    "ema_26": ema_26[-1] if ema_26 else None,
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": macd["signal"][-1] if macd["signal"] else None,
                    "bollinger_upper": bollinger["upper"][-1] if bollinger["upper"] else None,
                    "bollinger_lower": bollinger["lower"][-1] if bollinger["lower"] else None
                },
                "signals": signals,
                "trend": trend,
                "support_levels": support_resistance["support"][:3],  # Top 3 support levels
                "resistance_levels": support_resistance["resistance"][:3],  # Top 3 resistance levels
                "current_price": current_price,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"error": str(e)}
    
    async def compare(self, target_symbol: str, peers: List[str], benchmark: str) -> Dict[str, Any]:
        """Compare technical indicators across symbols"""
        # Placeholder for peer comparison
        return {
            "target": target_symbol,
            "peers": peers,
            "benchmark": benchmark,
            "comparison": "Technical comparison not implemented yet"
        }

class FundamentalAnalysisService(BaseAnalysisService):
    """Fundamental analysis service for financial data"""
    
    def __init__(self):
        super().__init__("FundamentalAnalysis")
    
    def calculate_financial_ratios(self, fundamentals: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key financial ratios"""
        ratios = {}
        
        # Valuation ratios
        ratios["pe_ratio"] = fundamentals.get("pe_ratio")
        ratios["forward_pe"] = fundamentals.get("forward_pe")
        ratios["peg_ratio"] = fundamentals.get("peg_ratio")
        ratios["price_to_book"] = fundamentals.get("price_to_book")
        ratios["price_to_sales"] = fundamentals.get("price_to_sales")
        ratios["ev_to_revenue"] = fundamentals.get("ev_to_revenue")
        ratios["ev_to_ebitda"] = fundamentals.get("ev_to_ebitda")
        
        # Profitability ratios
        ratios["profit_margin"] = fundamentals.get("profit_margin")
        ratios["operating_margin"] = fundamentals.get("operating_margin")
        ratios["return_on_assets"] = fundamentals.get("return_on_assets")
        ratios["return_on_equity"] = fundamentals.get("return_on_equity")
        
        # Growth ratios
        ratios["revenue_growth"] = fundamentals.get("revenue_growth")
        ratios["earnings_growth"] = fundamentals.get("earnings_growth")
        
        # Financial health ratios
        ratios["debt_to_equity"] = fundamentals.get("debt_to_equity")
        ratios["current_ratio"] = fundamentals.get("current_ratio")
        ratios["quick_ratio"] = fundamentals.get("quick_ratio")
        
        # Dividend ratios
        ratios["dividend_yield"] = fundamentals.get("dividend_yield")
        ratios["payout_ratio"] = fundamentals.get("payout_ratio")
        
        return {k: v for k, v in ratios.items() if v is not None}
    
    def assess_valuation(self, ratios: Dict[str, float]) -> Dict[str, Any]:
        """Assess company valuation based on ratios"""
        assessment = {
            "overall_valuation": "neutral",
            "valuation_score": 0,
            "factors": []
        }
        
        score = 0
        
        # PE ratio assessment
        pe_ratio = ratios.get("pe_ratio")
        if pe_ratio:
            if pe_ratio < 15:
                score += 1
                assessment["factors"].append("Low P/E ratio suggests undervaluation")
            elif pe_ratio > 25:
                score -= 1
                assessment["factors"].append("High P/E ratio suggests overvaluation")
        
        # PEG ratio assessment
        peg_ratio = ratios.get("peg_ratio")
        if peg_ratio:
            if peg_ratio < 1:
                score += 1
                assessment["factors"].append("PEG ratio < 1 suggests good value relative to growth")
            elif peg_ratio > 2:
                score -= 1
                assessment["factors"].append("PEG ratio > 2 suggests overvaluation relative to growth")
        
        # Price to book assessment
        pb_ratio = ratios.get("price_to_book")
        if pb_ratio:
            if pb_ratio < 1:
                score += 1
                assessment["factors"].append("P/B ratio < 1 suggests trading below book value")
            elif pb_ratio > 3:
                score -= 1
                assessment["factors"].append("P/B ratio > 3 suggests high premium to book value")
        
        # Determine overall valuation
        if score >= 2:
            assessment["overall_valuation"] = "undervalued"
        elif score <= -2:
            assessment["overall_valuation"] = "overvalued"
        
        assessment["valuation_score"] = score
        return assessment
    
    def assess_financial_health(self, ratios: Dict[str, float]) -> Dict[str, Any]:
        """Assess financial health based on ratios"""
        health = {
            "overall_health": "neutral",
            "health_score": 0,
            "strengths": [],
            "concerns": []
        }
        
        score = 0
        
        # Profitability assessment
        profit_margin = ratios.get("profit_margin")
        if profit_margin:
            if profit_margin > 0.15:  # 15%
                score += 1
                health["strengths"].append("Strong profit margins")
            elif profit_margin < 0.05:  # 5%
                score -= 1
                health["concerns"].append("Low profit margins")
        
        # Return on equity assessment
        roe = ratios.get("return_on_equity")
        if roe:
            if roe > 0.15:  # 15%
                score += 1
                health["strengths"].append("Strong return on equity")
            elif roe < 0.08:  # 8%
                score -= 1
                health["concerns"].append("Low return on equity")
        
        # Debt assessment
        debt_to_equity = ratios.get("debt_to_equity")
        if debt_to_equity:
            if debt_to_equity < 0.3:  # 30%
                score += 1
                health["strengths"].append("Low debt levels")
            elif debt_to_equity > 1.0:  # 100%
                score -= 1
                health["concerns"].append("High debt levels")
        
        # Liquidity assessment
        current_ratio = ratios.get("current_ratio")
        if current_ratio:
            if current_ratio > 2.0:
                score += 1
                health["strengths"].append("Strong liquidity position")
            elif current_ratio < 1.0:
                score -= 1
                health["concerns"].append("Potential liquidity concerns")
        
        # Determine overall health
        if score >= 2:
            health["overall_health"] = "strong"
        elif score <= -2:
            health["overall_health"] = "weak"
        
        health["health_score"] = score
        return health
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            fundamentals = data.get("fundamentals", {})
            if not fundamentals:
                return {"error": "No fundamental data available for analysis"}
            
            # Calculate financial ratios
            ratios = self.calculate_financial_ratios(fundamentals)
            
            if not ratios:
                return {"error": "Insufficient fundamental data for ratio calculation"}
            
            # Perform assessments
            valuation_assessment = self.assess_valuation(ratios)
            health_assessment = self.assess_financial_health(ratios)
            
            # Generate overall recommendation
            overall_score = valuation_assessment["valuation_score"] + health_assessment["health_score"]
            
            if overall_score >= 3:
                recommendation = "Strong Buy"
            elif overall_score >= 1:
                recommendation = "Buy"
            elif overall_score <= -3:
                recommendation = "Strong Sell"
            elif overall_score <= -1:
                recommendation = "Sell"
            else:
                recommendation = "Hold"
            
            return {
                "financial_ratios": ratios,
                "valuation_assessment": valuation_assessment,
                "financial_health": health_assessment,
                "recommendation": recommendation,
                "overall_score": overall_score,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return {"error": str(e)}
    
    async def compare(self, target_symbol: str, peers: List[str], benchmark: str) -> Dict[str, Any]:
        """Compare fundamental metrics across symbols"""
        # Placeholder for peer comparison
        return {
            "target": target_symbol,
            "peers": peers,
            "benchmark": benchmark,
            "comparison": "Fundamental comparison not implemented yet"
        }

class SentimentAnalysisService(BaseAnalysisService):
    """Sentiment analysis service for news and text data"""
    
    def __init__(self):
        super().__init__("SentimentAnalysis")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a text using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert polarity to sentiment label
            if polarity > 0.1:
                sentiment_label = "positive"
            elif polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "sentiment": sentiment_label,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "confidence": abs(polarity)
            }
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0, "confidence": 0}
    
    def extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from text"""
        financial_keywords = [
            "revenue", "profit", "earnings", "growth", "margin", "debt", "cash", "investment",
            "acquisition", "merger", "dividend", "buyback", "guidance", "outlook", "forecast",
            "bullish", "bearish", "rally", "decline", "volatility", "risk", "opportunity"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on news and text data"""
        try:
            news_data = data.get("news_data", {})
            articles = news_data.get("articles", [])
            
            if not articles:
                return {"error": "No news articles available for sentiment analysis"}
            
            sentiments = []
            overall_polarity = 0
            keyword_frequency = {}
            
            for article in articles:
                title = article.get("title", "")
                summary = article.get("summary", "")
                text = f"{title} {summary}"
                
                # Analyze sentiment
                sentiment_result = self.analyze_text_sentiment(text)
                sentiments.append({
                    "title": title,
                    "sentiment": sentiment_result["sentiment"],
                    "polarity": sentiment_result["polarity"],
                    "confidence": sentiment_result["confidence"]
                })
                
                overall_polarity += sentiment_result["polarity"]
                
                # Extract keywords
                keywords = self.extract_financial_keywords(text)
                for keyword in keywords:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
            
            # Calculate overall sentiment
            avg_polarity = overall_polarity / len(articles) if articles else 0
            
            if avg_polarity > 0.1:
                overall_sentiment = "positive"
            elif avg_polarity < -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Get top keywords
            top_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "overall_sentiment": overall_sentiment,
                "average_polarity": avg_polarity,
                "sentiment_distribution": {
                    "positive": len([s for s in sentiments if s["sentiment"] == "positive"]),
                    "negative": len([s for s in sentiments if s["sentiment"] == "negative"]),
                    "neutral": len([s for s in sentiments if s["sentiment"] == "neutral"])
                },
                "article_sentiments": sentiments,
                "top_keywords": top_keywords,
                "total_articles_analyzed": len(articles),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}

class RiskAnalysisService(BaseAnalysisService):
    """Risk analysis service for investment risk assessment"""
    
    def __init__(self):
        super().__init__("RiskAnalysis")
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if len(prices) < 2:
            return 0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def calculate_var(self, prices: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(prices) < 2:
            return 0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0
        
        peak = prices[0]
        max_dd = 0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_beta(self, asset_prices: List[float], market_prices: List[float]) -> float:
        """Calculate beta coefficient"""
        if len(asset_prices) != len(market_prices) or len(asset_prices) < 2:
            return 1.0  # Default beta
        
        asset_returns = [(asset_prices[i] - asset_prices[i-1]) / asset_prices[i-1] for i in range(1, len(asset_prices))]
        market_returns = [(market_prices[i] - market_prices[i-1]) / market_prices[i-1] for i in range(1, len(market_prices))]
        
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def assess_risk_factors(self, data: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        # Check fundamental risks
        fundamentals = data.get("fundamentals", {})
        if fundamentals:
            debt_to_equity = fundamentals.get("debt_to_equity")
            if debt_to_equity and debt_to_equity > 1.0:
                risk_factors.append("High debt-to-equity ratio")
            
            current_ratio = fundamentals.get("current_ratio")
            if current_ratio and current_ratio < 1.0:
                risk_factors.append("Liquidity concerns (current ratio < 1)")
            
            profit_margin = fundamentals.get("profit_margin")
            if profit_margin and profit_margin < 0:
                risk_factors.append("Negative profit margins")
        
        # Check technical risks
        historical_data = data.get("historical_data", {})
        if historical_data and "Close" in historical_data:
            prices = list(historical_data["Close"].values()) if isinstance(historical_data["Close"], dict) else list(historical_data["Close"])
            
            if len(prices) >= 20:
                volatility = self.calculate_volatility(prices)
                if volatility > 0.4:  # 40% annualized volatility
                    risk_factors.append("High price volatility")
                
                max_dd = self.calculate_max_drawdown(prices)
                if max_dd > 0.3:  # 30% max drawdown
                    risk_factors.append("Significant historical drawdowns")
        
        # Check sentiment risks
        sentiment_data = data.get("sentiment_analysis", {})
        if sentiment_data:
            overall_sentiment = sentiment_data.get("overall_sentiment")
            if overall_sentiment == "negative":
                risk_factors.append("Negative market sentiment")
        
        return risk_factors
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        try:
            historical_data = data.get("historical_data", {})
            
            if not historical_data or "Close" not in historical_data:
                return {"error": "No price data available for risk analysis"}
            
            prices = list(historical_data["Close"].values()) if isinstance(historical_data["Close"], dict) else list(historical_data["Close"])
            
            if len(prices) < 10:
                return {"error": "Insufficient price data for risk analysis"}
            
            # Calculate risk metrics
            volatility = self.calculate_volatility(prices)
            var_95 = self.calculate_var(prices, 0.95)
            var_99 = self.calculate_var(prices, 0.99)
            max_drawdown = self.calculate_max_drawdown(prices)
            
            # Calculate beta (using S&P 500 as proxy - in real implementation, fetch actual market data)
            beta = 1.0  # Placeholder
            
            # Assess risk level
            risk_score = 0
            if volatility > 0.3:
                risk_score += 2
            elif volatility > 0.2:
                risk_score += 1
            
            if max_drawdown > 0.2:
                risk_score += 2
            elif max_drawdown > 0.1:
                risk_score += 1
            
            if abs(var_95) > 0.05:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 4:
                risk_level = "High"
            elif risk_score >= 2:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Identify risk factors
            risk_factors = self.assess_risk_factors(data)
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "volatility": volatility,
                "var_95": var_95,
                "var_99": var_99,
                "max_drawdown": max_drawdown,
                "beta": beta,
                "risk_factors": risk_factors,
                "risk_mitigation_suggestions": [
                    "Diversify portfolio across sectors and asset classes",
                    "Consider position sizing based on volatility",
                    "Implement stop-loss orders",
                    "Monitor correlation with market indices",
                    "Regular portfolio rebalancing"
                ],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return {"error": str(e)}
    
    async def stress_test(self, data: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing under different scenarios"""
        try:
            market_change = scenario.get("market_change", 0)
            sector_change = scenario.get("sector_change", 0)
            
            # Simplified stress test calculation
            beta = data.get("beta", 1.0)
            current_price = data.get("current_price", 100)
            
            # Calculate expected price change
            expected_change = (beta * market_change) + (0.5 * sector_change)
            stressed_price = current_price * (1 + expected_change)
            
            return {
                "scenario": scenario["name"],
                "current_price": current_price,
                "stressed_price": stressed_price,
                "price_change": expected_change,
                "expected_return": expected_change,
                "risk_score": abs(expected_change) * 10  # Simple risk scoring
            }
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {"error": str(e)}