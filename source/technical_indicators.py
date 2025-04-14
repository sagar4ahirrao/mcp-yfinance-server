from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import yfinance as yf

class TechnicalIndicators:
    """
    Class that provides various technical indicators and analysis tools for stock data.
    """
    
    @staticmethod
    def get_stock_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve historical stock data for technical analysis.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            return data
        except Exception as e:
            raise ValueError(f"Error retrieving data for {symbol}: {e}")
    
    @staticmethod
    def calculate_moving_average(data: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate simple moving average.
        
        Args:
            data: DataFrame with price data
            window: Period for moving average
            column: Column name to calculate MA for (default: Close)
            
        Returns:
            Series with moving average values
        """
        return data[column].rolling(window=window).mean()
    
    @staticmethod
    def calculate_exponential_moving_average(data: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate exponential moving average.
        
        Args:
            data: DataFrame with price data
            window: Period for EMA
            column: Column name to calculate EMA for (default: Close)
            
        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            window: RSI period (default: 14)
            column: Column name to calculate RSI for (default: Close)
            
        Returns:
            Series with RSI values
        """
        delta = data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate first RSI after initial averaging period
        for i in range(window, len(delta)):
            if i > window:  # Use EMA for subsequent calculations
                avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
                avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                      signal_period: int = 9, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            column: Column name to calculate MACD for (default: Close)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        fast_ema = TechnicalIndicators.calculate_exponential_moving_average(data, fast_period, column)
        slow_ema = TechnicalIndicators.calculate_exponential_moving_average(data, slow_period, column)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, 
                                num_std: float = 2.0, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            window: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            column: Column name for calculation (default: Close)
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands as Series
        """
        middle_band = TechnicalIndicators.calculate_moving_average(data, window, column)
        std_dev = data[column].rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with price data
            window: ATR period (default: 14)
            
        Returns:
            Series with ATR values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20, column: str = 'Close', 
                           annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: DataFrame with price data
            window: Period for volatility calculation (default: 20)
            column: Column name to calculate volatility for (default: Close)
            annualize: Whether to annualize the volatility (default: True)
            
        Returns:
            Series with volatility values
        """
        # Calculate logarithmic returns
        log_returns = np.log(data[column] / data[column].shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=window).std()
        
        # Annualize if requested (assuming 252 trading days)
        if annualize:
            if 'd' in data.index.freq or data.index.freq is None:  # Daily data
                volatility = volatility * np.sqrt(252)
            elif 'h' in data.index.freq:  # Hourly data
                volatility = volatility * np.sqrt(252 * 6.5)  # ~6.5 trading hours per day
            elif 'm' in data.index.freq:  # Minute data
                volatility = volatility * np.sqrt(252 * 6.5 * 60)
                
        return volatility
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20, 
                               sensitivity: float = 0.03) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using local minima and maxima.
        
        Args:
            data: DataFrame with price data
            window: Lookback period for finding pivots (default: 20)
            sensitivity: Minimum price change percentage to consider (default: 0.03)
            
        Returns:
            Dictionary with 'support' and 'resistance' levels
        """
        high = data['High']
        low = data['Low']
        
        resistance_levels = []
        support_levels = []
        
        # Find pivot highs (local maxima)
        for i in range(window, len(high) - window):
            if all(high[i] > high[i-j] for j in range(1, window+1)) and all(high[i] > high[i+j] for j in range(1, window+1)):
                # Check if significantly different from previously found resistance levels
                if not any(abs(high[i] - level) / level < sensitivity for level in resistance_levels):
                    resistance_levels.append(high[i])
                    
        # Find pivot lows (local minima)
        for i in range(window, len(low) - window):
            if all(low[i] < low[i-j] for j in range(1, window+1)) and all(low[i] < low[i+j] for j in range(1, window+1)):
                # Check if significantly different from previously found support levels
                if not any(abs(low[i] - level) / level < sensitivity for level in support_levels):
                    support_levels.append(low[i])
                    
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels)
        }
    
    @staticmethod
    def detect_trends(data: pd.DataFrame, short_window: int = 20, long_window: int = 50, 
                    column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Detect trends using moving average crossovers.
        
        Args:
            data: DataFrame with price data
            short_window: Short-term MA period (default: 20)
            long_window: Long-term MA period (default: 50)
            column: Column name to detect trends for (default: Close)
            
        Returns:
            Dictionary with 'trend' and 'signal' Series
        """
        short_ma = TechnicalIndicators.calculate_moving_average(data, short_window, column)
        long_ma = TechnicalIndicators.calculate_moving_average(data, long_window, column)
        
        # Create trend indicator (1: uptrend, -1: downtrend, 0: neutral/undefined)
        trend = pd.Series(0, index=data.index)
        trend[short_ma > long_ma] = 1  # Uptrend
        trend[short_ma < long_ma] = -1  # Downtrend
        
        # Create signal for trend changes
        signal = pd.Series(0, index=data.index)
        signal[(trend.shift(1) <= 0) & (trend > 0)] = 1  # Buy signal (trend turning positive)
        signal[(trend.shift(1) >= 0) & (trend < 0)] = -1  # Sell signal (trend turning negative)
        
        return {
            'trend': trend,
            'signal': signal
        }
    
    @staticmethod
    def calculate_pattern_recognition(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Basic pattern recognition for common candlestick patterns.
        
        Args:
            data: DataFrame with price data (must have Open, High, Low, Close)
            
        Returns:
            Dictionary with pattern signals (1 where pattern is detected)
        """
        pattern_signals = {}
        
        # Doji pattern (open and close are very close)
        doji = pd.Series(0, index=data.index)
        body_size = abs(data['Close'] - data['Open'])
        avg_body = body_size.rolling(window=14).mean()
        shadow_size = data['High'] - data['Low']
        doji[(body_size < 0.1 * shadow_size) & (body_size < 0.25 * avg_body)] = 1
        pattern_signals['doji'] = doji
        
        # Hammer pattern (long lower shadow, small body at the top)
        hammer = pd.Series(0, index=data.index)
        lower_shadow = pd.Series(0, index=data.index)
        upper_shadow = pd.Series(0, index=data.index)
        
        # For days with close > open (bullish)
        bullish = data['Close'] > data['Open']
        lower_shadow[bullish] = data['Open'][bullish] - data['Low'][bullish]
        upper_shadow[bullish] = data['High'][bullish] - data['Close'][bullish]
        
        # For days with open > close (bearish)
        bearish = data['Open'] > data['Close']
        lower_shadow[bearish] = data['Close'][bearish] - data['Low'][bearish]
        upper_shadow[bearish] = data['High'][bearish] - data['Open'][bearish]
        
        # Hammer criteria
        body_height = abs(data['Close'] - data['Open'])
        hammer[(lower_shadow > 2 * body_height) & (upper_shadow < 0.2 * body_height)] = 1
        pattern_signals['hammer'] = hammer
        
        # Engulfing pattern (current candle completely engulfs previous candle)
        bullish_engulfing = pd.Series(0, index=data.index)
        bearish_engulfing = pd.Series(0, index=data.index)
        
        # Bullish engulfing
        bullish_engulfing[(data['Open'] < data['Close'].shift(1)) & 
                         (data['Close'] > data['Open'].shift(1)) &
                         (data['Close'] > data['Open']) &
                         (data['Open'].shift(1) > data['Close'].shift(1))] = 1
        
        # Bearish engulfing
        bearish_engulfing[(data['Open'] > data['Close'].shift(1)) & 
                         (data['Close'] < data['Open'].shift(1)) &
                         (data['Close'] < data['Open']) &
                         (data['Open'].shift(1) < data['Close'].shift(1))] = 1
        
        pattern_signals['bullish_engulfing'] = bullish_engulfing
        pattern_signals['bearish_engulfing'] = bearish_engulfing
        
        return pattern_signals
    
    @staticmethod
    def detect_divergence(data: pd.DataFrame, indicator: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Detect divergence between price and indicator (e.g., RSI).
        
        Args:
            data: DataFrame with price data
            indicator: Series with indicator values (e.g., RSI)
            window: Lookback period for finding pivots (default: 14)
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' Series
        """
        close = data['Close']
        
        bullish_divergence = pd.Series(0, index=data.index)
        bearish_divergence = pd.Series(0, index=data.index)
        
        # Find local price lows and indicator lows
        for i in range(window, len(close) - window):
            # Check for price making lower low
            if (close[i] < close[i-1]) and (close[i] < close[i+1]) and \
               (close[i] < min(close[i-window:i])) and (close[i] < min(close[i+1:i+window+1])):
                
                # But indicator making higher low (bullish divergence)
                if (indicator[i] > indicator[i-window]) and (indicator[i] > indicator[i-window//2]):
                    bullish_divergence[i] = 1
        
        # Find local price highs and indicator highs
        for i in range(window, len(close) - window):
            # Check for price making higher high
            if (close[i] > close[i-1]) and (close[i] > close[i+1]) and \
               (close[i] > max(close[i-window:i])) and (close[i] > max(close[i+1:i+window+1])):
                
                # But indicator making lower high (bearish divergence)
                if (indicator[i] < indicator[i-window]) and (indicator[i] < indicator[i-window//2]):
                    bearish_divergence[i] = 1
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }


# Register the technical indicators with the MCP server
def register_technical_indicators(mcp: FastMCP):
    """
    Register all technical indicator tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance
    """
    ti = TechnicalIndicators()
    
    @mcp.tool()
    def get_moving_averages(symbol: str, period: str = "6mo", interval: str = "1d", 
                           windows: List[int] = [20, 50, 200]) -> Dict[str, List[float]]:
        """
        Calculate multiple moving averages for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (e.g., "6mo", "1y", "max")
            interval: Data interval (e.g., "1d", "1wk")
            windows: List of MA periods to calculate
            
        Returns:
            Dictionary with moving average values
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            result = {}
            
            for window in windows:
                ma = ti.calculate_moving_average(data, window)
                ema = ti.calculate_exponential_moving_average(data, window)
                
                result[f'SMA_{window}'] = ma.dropna().tolist()
                result[f'EMA_{window}'] = ema.dropna().tolist()
                
            # Also include dates for reference
            result['dates'] = data.index.strftime('%Y-%m-%d').tolist()
            result['close'] = data['Close'].tolist()
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_rsi(symbol: str, period: str = "6mo", interval: str = "1d", 
               window: int = 14) -> Dict[str, List[float]]:
        """
        Calculate RSI for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            window: RSI period
            
        Returns:
            Dictionary with RSI values and dates
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            rsi = ti.calculate_rsi(data, window)
            
            return {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'rsi': rsi.dropna().tolist(),
                'close': data['Close'].tolist()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_macd(symbol: str, period: str = "6mo", interval: str = "1d", 
                fast_period: int = 12, slow_period: int = 26, 
                signal_period: int = 9) -> Dict[str, List[float]]:
        """
        Calculate MACD for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with MACD values and dates
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            macd_data = ti.calculate_macd(data, fast_period, slow_period, signal_period)
            
            return {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'macd': macd_data['macd'].dropna().tolist(),
                'signal': macd_data['signal'].dropna().tolist(),
                'histogram': macd_data['histogram'].dropna().tolist(),
                'close': data['Close'].tolist()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_bollinger_bands(symbol: str, period: str = "6mo", interval: str = "1d",
                           window: int = 20, num_std: float = 2.0) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            window: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with Bollinger Bands values and dates
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            bb_data = ti.calculate_bollinger_bands(data, window, num_std)
            
            return {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'upper': bb_data['upper'].dropna().tolist(),
                'middle': bb_data['middle'].dropna().tolist(),
                'lower': bb_data['lower'].dropna().tolist(),
                'close': data['Close'].tolist()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_volatility_analysis(symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, List[float]]:
        """
        Calculate volatility metrics for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with volatility metrics and dates
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            
            # Calculate various volatility metrics
            vol_20d = ti.calculate_volatility(data, window=20)
            vol_50d = ti.calculate_volatility(data, window=50)
            atr = ti.calculate_atr(data)
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            
            return {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'volatility_20d': vol_20d.dropna().tolist(),
                'volatility_50d': vol_50d.dropna().tolist(),
                'atr': atr.dropna().tolist(),
                'daily_returns': data['Returns'].dropna().tolist(),
                'close': data['Close'].tolist()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_support_resistance(symbol: str, period: str = "1y", interval: str = "1d",
                              window: int = 20) -> Dict[str, List[float]]:
        """
        Find support and resistance levels for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            window: Lookback period for pivot points
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            levels = ti.detect_support_resistance(data, window)
            
            # Get the latest close for context
            latest_close = data['Close'].iloc[-1]
            
            return {
                'support_levels': levels['support'],
                'resistance_levels': levels['resistance'],
                'latest_close': float(latest_close)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_trend_analysis(symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, any]:
        """
        Complete trend analysis for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            data = ti.get_stock_data(symbol, period, interval)
            
            # Calculate various trend indicators
            trends = ti.detect_trends(data)
            patterns = ti.calculate_pattern_recognition(data)
            rsi = ti.calculate_rsi(data)
            
            # Detect divergence between price and RSI
            divergences = ti.detect_divergence(data, rsi)
            
            # Filter signals to the last 10 days
            last_10_days = -10
            
            # Compile signals
            signals = []
            dates = data.index[last_10_days:].strftime('%Y-%m-%d').tolist()
            
            for i, date in enumerate(dates):
                idx = i + len(data) + last_10_days
                if idx >= len(data):
                    continue
                    
                day_signals = []
                
                # Check for trend changes
                if trends['signal'].iloc[idx] == 1:
                    day_signals.append("Bullish trend change")
                elif trends['signal'].iloc[idx] == -1:
                    day_signals.append("Bearish trend change")
                
                # Check for patterns
                for pattern, signal in patterns.items():
                    if signal.iloc[idx] == 1:
                        day_signals.append(f"{pattern.replace('_', ' ').title()} pattern")
                
                # Check for divergences
                if divergences['bullish_divergence'].iloc[idx] == 1:
                    day_signals.append("Bullish divergence")
                elif divergences['bearish_divergence'].iloc[idx] == 1:
                    day_signals.append("Bearish divergence")
                
                if day_signals:
                    signals.append({
                        'date': date,
                        'signals': day_signals
                    })
            
            # Determine overall trend
            latest_trend = trends['trend'].iloc[-1]
            if latest_trend > 0:
                overall_trend = "Bullish"
            elif latest_trend < 0:
                overall_trend = "Bearish"
            else:
                overall_trend = "Neutral"
            
            return {
                'overall_trend': overall_trend,
                'signals': signals,
                'trends': trends['trend'].iloc[last_10_days:].tolist(),
                'close': data['Close'].iloc[last_10_days:].tolist(),
                'dates': dates
            }
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def get_technical_summary(symbol: str) -> Dict[str, any]:
        """
        Generate a complete technical analysis summary for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with technical analysis summary
        """
        try:
            # Get data with different timeframes
            data_daily = ti.get_stock_data(symbol, period="6mo", interval="1d")
            data_weekly = ti.get_stock_data(symbol, period="2y", interval="1wk")
            latest_price = data_daily['Close'].iloc[-1]
            
            # Calculate indicators
            sma_20 = ti.calculate_moving_average(data_daily, 20).iloc[-1]
            sma_50 = ti.calculate_moving_average(data_daily, 50).iloc[-1]
            sma_200 = ti.calculate_moving_average(data_daily, 200).iloc[-1]
            
            ema_12 = ti.calculate_exponential_moving_average(data_daily, 12).iloc[-1]
            ema_26 = ti.calculate_exponential_moving_average(data_daily, 26).iloc[-1]
            
            rsi_14 = ti.calculate_rsi(data_daily).iloc[-1]
            
            macd_data = ti.calculate_macd(data_daily)
            macd = macd_data['macd'].iloc[-1]
            macd_signal = macd_data['signal'].iloc[-1]
            
            bb_data = ti.calculate_bollinger_bands(data_daily)
            bb_upper = bb_data['upper'].iloc[-1]
            bb_lower = bb_data['lower'].iloc[-1]
            
            volatility = ti.calculate_volatility(data_daily).iloc[-1]
            
            # Support and resistance
            levels = ti.detect_support_resistance(data_daily)
            supports = [level for level in levels['support'] if level < latest_price]
            resistances = [level for level in levels['resistance'] if level > latest_price]
            nearest_support = max(supports) if supports else None
            nearest_resistance = min(resistances) if resistances else None
            
            # Trend analysis
            daily_trend = ti.detect_trends(data_daily)['trend'].iloc[-1]
            weekly_trend = ti.detect_trends(data_weekly)['trend'].iloc[-1]
            
            # Generate signals
            signals = []
            
            # Moving average signals
            if latest_price > sma_20:
                signals.append("Price above SMA(20) - short-term bullish")
            else:
                signals.append("Price below SMA(20) - short-term bearish")
                
            if latest_price > sma_50:
                signals.append("Price above SMA(50) - medium-term bullish")
            else:
                signals.append("Price below SMA(50) - medium-term bearish")
                
            if latest_price > sma_200:
                signals.append("Price above SMA(200) - long-term bullish")
            else:
                signals.append("Price below SMA(200) - long-term bearish")
                
            # Golden/Death cross
            if sma_50 > sma_200 and sma_50.shift(1) <= sma_200.shift(1):
                signals.append("Recent Golden Cross (SMA50 crossed above SMA200) - major bullish signal")
            if sma_50 < sma_200 and sma_50.shift(1) >= sma_200.shift(1):
                signals.append("Recent Death Cross (SMA50 crossed below SMA200) - major bearish signal")
                
              # RSI signals
            if rsi_14 > 70:
                signals.append("RSI above 70 - overbought condition")
            elif rsi_14 < 30:
                signals.append("RSI below 30 - oversold condition")
                
            # MACD signals
            if macd > macd_signal and macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]:
                signals.append("MACD bullish crossover - buy signal")
            elif macd < macd_signal and macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2]:
                signals.append("MACD bearish crossover - sell signal")
                
            # Bollinger Bands signals
            if latest_price > bb_upper:
                signals.append("Price above upper Bollinger Band - overbought/strong trend")
            elif latest_price < bb_lower:
                signals.append("Price below lower Bollinger Band - oversold/strong trend")
                
            # Bollinger Band squeeze (low volatility, potential breakout)
            band_width = (bb_upper - bb_lower) / bb_middle
            avg_band_width = ((data_daily['High'] - data_daily['Low']) / data_daily['Close']).rolling(20).mean().iloc[-1]
            
            if band_width < 0.7 * avg_band_width:
                signals.append("Bollinger Band squeeze - low volatility, potential breakout")
                
            # Determine overall bias based on multiple timeframes
            if daily_trend > 0 and weekly_trend > 0:
                overall_bias = "Strong Bullish"
            elif daily_trend > 0 and weekly_trend <= 0:
                overall_bias = "Moderately Bullish"
            elif daily_trend <= 0 and weekly_trend > 0:
                overall_bias = "Neutral with Bullish Bias"
            else:
                overall_bias = "Bearish"
                
            # Format results for return
            return {
                'symbol': symbol,
                'last_price': float(latest_price),
                'overall_bias': overall_bias,
                'signals': signals,
                'indicators': {
                    'sma_20': float(sma_20),
                    'sma_50': float(sma_50), 
                    'sma_200': float(sma_200),
                    'ema_12': float(ema_12),
                    'ema_26': float(ema_26),
                    'rsi_14': float(rsi_14),
                    'macd': float(macd),
                    'macd_signal': float(macd_signal),
                    'bb_upper': float(bb_upper),
                    'bb_middle': float(bb_middle),
                    'bb_lower': float(bb_lower),
                    'volatility_annualized': float(volatility * 100)  # Convert to percentage
                },
                'support_resistance': {
                    'nearest_support': float(nearest_support) if nearest_support else None,
                    'nearest_resistance': float(nearest_resistance) if nearest_resistance else None
                }
            }
        except Exception as e:
            return {"error": str(e)}