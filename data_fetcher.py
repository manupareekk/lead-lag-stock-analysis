import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Handles fetching and caching stock data from Yahoo Finance"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        """Generate cache file path for given parameters"""
        filename = f"{symbol}_{period}_{interval}.pkl"
        return self.cache_dir / filename
    
    def is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    def fetch_stock_data(self, 
                        symbol: str, 
                        period: str = "2y", 
                        interval: str = "1d",
                        use_cache: bool = True) -> pd.DataFrame:
        """Fetch stock data with caching support
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_path = self.get_cache_path(symbol, period, interval)
        
        # Try to load from cache first
        if use_cache and self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {symbol} from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        
        # Fetch fresh data
        try:
            logger.info(f"Fetching {symbol} data from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Cache the data
            if use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Cached {symbol} data")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
    
    def fetch_multiple_stocks(self, 
                             symbols: List[str], 
                             period: str = "2y", 
                             interval: str = "1d",
                             delay: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks with rate limiting
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            delay: Delay between requests to avoid rate limiting
        
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                results[symbol] = data
                time.sleep(delay)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        return results
    
    def get_price_data(self, 
                      symbols: List[str], 
                      column: str = 'Close',
                      period: str = "2y", 
                      interval: str = "1d") -> pd.DataFrame:
        """Get specific price column for multiple stocks as a DataFrame
        
        Args:
            symbols: List of stock symbols
            column: Price column to extract ('Open', 'High', 'Low', 'Close', 'Volume')
            period: Data period
            interval: Data interval
        
        Returns:
            DataFrame with symbols as columns and dates as index
        """
        stock_data = self.fetch_multiple_stocks(symbols, period, interval)
        
        price_data = pd.DataFrame()
        for symbol, data in stock_data.items():
            if not data.empty and column in data.columns:
                price_data[symbol] = data[column]
        
        return price_data.dropna()
    
    def clear_cache(self, symbol: str = None):
        """Clear cache files
        
        Args:
            symbol: If provided, clear cache only for this symbol. Otherwise clear all.
        """
        if symbol:
            cache_files = list(self.cache_dir.glob(f"{symbol}_*.pkl"))
        else:
            cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            cache_file.unlink()
            logger.info(f"Removed cache file: {cache_file}")
    
    def calculate_returns(self, price_data: pd.DataFrame, method: str = 'pct_change') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            price_data: DataFrame with price data
            method: Method to calculate returns ('pct_change', 'log_returns')
        
        Returns:
            DataFrame with returns
        """
        if method == 'pct_change':
            returns = price_data.pct_change()
        elif method == 'log_returns':
            returns = np.log(price_data / price_data.shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
        
        return returns.dropna()
    
    def detrend_data(self, data: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        Remove trend from data
        
        Args:
            data: DataFrame with data to detrend
            method: Detrending method ('linear', 'quadratic', 'moving_average')
        
        Returns:
            Detrended DataFrame
        """
        from scipy import signal
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        detrended_data = data.copy()
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 10:  # Need minimum data points
                continue
                
            if method == 'linear':
                # Linear detrending
                detrended_series = signal.detrend(series.values, type='linear')
                detrended_data.loc[series.index, column] = detrended_series
                
            elif method == 'quadratic':
                # Quadratic detrending
                X = np.arange(len(series)).reshape(-1, 1)
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X)
                
                reg = LinearRegression().fit(X_poly, series.values)
                trend = reg.predict(X_poly)
                
                detrended_data.loc[series.index, column] = series.values - trend
                
            elif method == 'moving_average':
                # Remove moving average trend
                window = min(20, len(series) // 4)  # Adaptive window size
                ma = series.rolling(window=window, center=True).mean()
                detrended_data.loc[series.index, column] = series - ma
                
        return detrended_data.dropna()
    
    def remove_market_effects(self, data: pd.DataFrame, market_index: str = 'SPY',
                            method: str = 'subtract') -> pd.DataFrame:
        """
        Remove market-wide effects from stock data
        
        Args:
            data: DataFrame with stock data
            market_index: Market index symbol to use as benchmark
            method: Method to remove market effects ('subtract', 'beta_adjust', 'residual')
        
        Returns:
            Market-adjusted DataFrame
        """
        try:
            # Get market index data
            market_data = self.fetch_stock_data(
                market_index, 
                period='2y',  # Get enough data
                interval='1d'
            )
            
            if market_data is None or market_data.empty:
                logger.warning(f"Could not fetch market index {market_index}")
                return data
            
            # Calculate market returns
            market_returns = self.calculate_returns(market_data[['Close']], method='pct_change')
            market_returns = market_returns['Close']
            
            # Calculate stock returns if not already returns
            if data.max().max() > 10:  # Assume it's price data if values > 10
                stock_returns = self.calculate_returns(data, method='pct_change')
            else:
                stock_returns = data.copy()
            
            # Align dates
            common_dates = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
            
            adjusted_data = stock_returns.copy()
            
            for column in stock_returns.columns:
                stock_series = stock_returns[column].dropna()
                market_aligned = market_returns.loc[stock_series.index]
                
                # Remove any remaining NaN values
                valid_idx = stock_series.index.intersection(market_aligned.dropna().index)
                if len(valid_idx) < 30:  # Need minimum observations
                    continue
                    
                stock_clean = stock_series.loc[valid_idx]
                market_clean = market_aligned.loc[valid_idx]
                
                if method == 'subtract':
                    # Simple market return subtraction
                    adjusted_data.loc[valid_idx, column] = stock_clean - market_clean
                    
                elif method == 'beta_adjust':
                    # Beta-adjusted returns: stock_return - beta * market_return
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(market_clean, stock_clean)
                    beta = slope
                    adjusted_data.loc[valid_idx, column] = stock_clean - beta * market_clean
                    
                elif method == 'residual':
                    # Use regression residuals
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression().fit(market_clean.values.reshape(-1, 1), stock_clean.values)
                    predicted = reg.predict(market_clean.values.reshape(-1, 1))
                    residuals = stock_clean.values - predicted
                    adjusted_data.loc[valid_idx, column] = residuals
            
            return adjusted_data.dropna()
            
        except Exception as e:
            logger.error(f"Market adjustment failed: {e}")
            return data
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize data for cross-stock comparisons
        
        Args:
            data: DataFrame to normalize
            method: Normalization method ('zscore', 'minmax', 'robust')
        
        Returns:
            Normalized DataFrame
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        normalized_data = data.copy()
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) > 0:
                scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
                normalized_data.loc[series.index, column] = scaled_values
        
        return normalized_data

# Example usage
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # Test fetching single stock
    aapl_data = fetcher.fetch_stock_data("AAPL")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.head())
    
    # Test fetching multiple stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "META"]
    price_data = fetcher.get_price_data(symbols)
    print(f"\nPrice data shape: {price_data.shape}")
    print(price_data.head())