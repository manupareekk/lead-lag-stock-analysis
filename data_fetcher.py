import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
from pathlib import Path
import pickle
import asyncio
import aiohttp
import concurrent.futures
from functools import lru_cache
import threading
from queue import Queue
import warnings
from performance_monitor import performance_monitor
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Handles fetching and caching stock data from Yahoo Finance"""
    
    def __init__(self, cache_dir: str = "cache", max_workers: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self._session_cache = {}
        self._lock = threading.Lock()
        
    def get_cache_path(self, symbol: str, period: str, interval: str) -> Path:
        """Generate cache file path for given parameters"""
        filename = f"{symbol}_{period}_{interval}.pkl"
        return self.cache_dir / filename
    
    def is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    @lru_cache(maxsize=1000)
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """LRU cached method for loading data from disk cache"""
        try:
            cache_path = Path(cache_key)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached data from {cache_key}: {e}")
        return None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'memory_cache_size': self._get_cached_data.cache_info().currsize,
            'memory_cache_hits': self._get_cached_data.cache_info().hits,
            'memory_cache_misses': self._get_cached_data.cache_info().misses
        }
    
    def optimize_cache(self, max_age_days: int = 7):
        """Remove old cache files to optimize storage"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        removed_count = 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for cache_file in cache_files:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                cache_file.unlink()
                removed_count += 1
        
        # Clear memory cache
        self._get_cached_data.cache_clear()
        
        logger.info(f"Removed {removed_count} old cache files (older than {max_age_days} days)")
        return removed_count
    
    def validate_period_interval(self, period: str, interval: str) -> Tuple[str, str]:
        """Validate and adjust period-interval combinations based on Yahoo Finance limits
        
        Args:
            period: Requested data period
            interval: Requested data interval
            
        Returns:
            Tuple of (validated_period, validated_interval)
        """
        # Yahoo Finance API limitations
        if interval == "1m":
            # 1-minute data is limited to 7 days
            if period not in ["1d", "2d", "5d", "7d"]:
                logger.warning(f"1m interval limited to 7 days max. Adjusting period from {period} to 7d")
                period = "7d"
        elif interval in ["5m", "15m", "30m"]:
            # Sub-hourly data: maximum 60 days
            if period in ["1w", "1mo", "6mo", "1y", "2y", "5y", "max"]:
                if period == "1w":
                    logger.warning(f"Period '{period}' adjusted for {interval} interval. Using '7d'.")
                    period = "7d"
                elif period == "1mo":
                    logger.warning(f"Period '{period}' adjusted for {interval} interval. Using '30d'.")
                    period = "30d"
                else:
                    logger.warning(f"Period '{period}' too long for {interval} interval. Adjusting to '60d'.")
                    period = "60d"
        elif interval == "1h":
            # 1-hour data is limited to 730 days (about 2 years)
            long_periods = ["5y", "10y", "max"]
            if period in long_periods:
                logger.warning(f"1h interval limited to ~2 years. Adjusting period from {period} to 2y")
                period = "2y"
        
        return period, interval
    
    def fetch_stock_data(self, 
                        symbol: str, 
                        period: str = "2y", 
                        interval: str = "1d",
                        use_cache: bool = True) -> pd.DataFrame:
        """Fetch stock data with caching support
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1w', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        # Validate and adjust period-interval combination
        period, interval = self.validate_period_interval(period, interval)
        
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
    
    def fetch_multiple_stocks_concurrent(self, 
                                       symbols: List[str], 
                                       period: str = "2y", 
                                       interval: str = "1d",
                                       max_workers: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks concurrently for faster performance
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            max_workers: Maximum number of concurrent workers (defaults to class setting)
        
        Returns:
            Dictionary mapping symbols to their data
        """
        start_time = time.time()
        if max_workers is None:
            max_workers = self.max_workers
            
        results = {}
        
        # First, check cache for all symbols to avoid unnecessary API calls
        uncached_symbols = []
        cached_count = 0
        for symbol in symbols:
            period_adj, interval_adj = self.validate_period_interval(period, interval)
            cache_path = self.get_cache_path(symbol, period_adj, interval_adj)
            
            if self.is_cache_valid(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    results[symbol] = data
                    cached_count += 1
                    logger.info(f"Loaded {symbol} from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cache for {symbol}: {e}")
                    uncached_symbols.append(symbol)
            else:
                uncached_symbols.append(symbol)
        
        # Fetch uncached symbols concurrently
        if uncached_symbols:
            logger.info(f"Fetching {len(uncached_symbols)} symbols concurrently with {max_workers} workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fetch tasks
                future_to_symbol = {
                    executor.submit(self.fetch_stock_data, symbol, period, interval): symbol 
                    for symbol in uncached_symbols
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result(timeout=30)  # 30 second timeout per symbol
                        results[symbol] = data
                    except Exception as e:
                        logger.error(f"Failed to fetch {symbol}: {e}")
                        continue
        
        # Record performance metrics
        fetch_time = time.time() - start_time
        total_data_points = sum(len(df) for df in results.values() if not df.empty)
        
        performance_monitor.record_fetch(
            symbols=symbols,
            fetch_time=fetch_time,
            cache_hits=cached_count,
            cache_misses=len(uncached_symbols),
            method_used="concurrent_fetch",
            data_points=total_data_points
        )
        
        logger.info(f"Concurrent fetch completed: {len(results)}/{len(symbols)} symbols in {fetch_time:.2f}s")
        return results
    
    def fetch_batch_yfinance(self, symbols: List[str], period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch multiple symbols in a single yfinance call for maximum efficiency
        
        Args:
            symbols: List of stock symbols (max 10-15 for best performance)
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to their data
        """
        if len(symbols) > 15:
            logger.warning(f"Batch size {len(symbols)} is large, consider splitting for better performance")
        
        start_time = time.time()
        period, interval = self.validate_period_interval(period, interval)
        
        try:
            # Use yfinance download for batch fetching
            logger.info(f"Batch fetching {len(symbols)} symbols: {', '.join(symbols)}")
            
            # Create ticker string
            ticker_string = ' '.join(symbols)
            
            # Fetch all data at once
            batch_data = yf.download(
                ticker_string,
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False
            )
            
            results = {}
            
            if len(symbols) == 1:
                # Single symbol case
                symbol = symbols[0]
                if not batch_data.empty:
                    results[symbol] = batch_data
                    # Cache the result
                    cache_path = self.get_cache_path(symbol, period, interval)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(batch_data, f)
            else:
                # Multiple symbols case
                for symbol in symbols:
                    try:
                        if symbol in batch_data.columns.levels[0]:
                            symbol_data = batch_data[symbol]
                            if not symbol_data.empty:
                                results[symbol] = symbol_data
                                # Cache individual results
                                cache_path = self.get_cache_path(symbol, period, interval)
                                with open(cache_path, 'wb') as f:
                                    pickle.dump(symbol_data, f)
                                logger.info(f"Cached {symbol} data")
                    except Exception as e:
                        logger.warning(f"Failed to process {symbol} from batch: {e}")
                        continue
            
            # Record performance metrics
            fetch_time = time.time() - start_time
            total_data_points = sum(len(df) for df in results.values() if not df.empty)
            
            performance_monitor.record_fetch(
                symbols=symbols,
                fetch_time=fetch_time,
                cache_hits=0,  # No cache hits in this method
                cache_misses=len(symbols),
                method_used="batch_yfinance",
                data_points=total_data_points
            )
            
            logger.info(f"Successfully fetched {len(results)} out of {len(symbols)} symbols in {fetch_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fallback to individual fetching
            logger.info("Falling back to individual symbol fetching")
            return self.fetch_multiple_stocks_concurrent(symbols, period, interval)
    
    def smart_fetch_multiple_stocks(self, 
                                  symbols: List[str], 
                                  period: str = "2y", 
                                  interval: str = "1d",
                                  batch_size: int = 10) -> Dict[str, pd.DataFrame]:
        """Intelligently fetch multiple stocks using optimal strategy
        
        This method combines caching, batching, and concurrent fetching for maximum performance.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            batch_size: Size of batches for batch fetching
        
        Returns:
            Dictionary mapping symbols to their data
        """
        start_time = time.time()
        logger.info(f"Smart fetching {len(symbols)} symbols with batch size {batch_size}")
        
        results = {}
        remaining_symbols = symbols.copy()
        
        # Step 1: Load from cache first
        cached_symbols = []
        for symbol in symbols:
            period_adj, interval_adj = self.validate_period_interval(period, interval)
            cache_path = self.get_cache_path(symbol, period_adj, interval_adj)
            
            if self.is_cache_valid(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    results[symbol] = data
                    cached_symbols.append(symbol)
                    logger.debug(f"Loaded {symbol} from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cache for {symbol}: {e}")
        
        # Remove cached symbols from remaining
        remaining_symbols = [s for s in remaining_symbols if s not in cached_symbols]
        logger.info(f"Loaded {len(cached_symbols)} symbols from cache, {len(remaining_symbols)} remaining")
        
        # Step 2: Batch fetch remaining symbols
        if remaining_symbols:
            # Split into batches
            batches = [remaining_symbols[i:i + batch_size] for i in range(0, len(remaining_symbols), batch_size)]
            
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} symbols")
                
                try:
                    batch_results = self.fetch_batch_yfinance(batch, period, interval)
                    results.update(batch_results)
                    
                    # Small delay between batches to be respectful to the API
                    if i < len(batches) - 1:  # Don't sleep after the last batch
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Batch {i+1} failed: {e}")
                    # Fallback to concurrent individual fetching for this batch
                    logger.info(f"Falling back to concurrent fetching for batch {i+1}")
                    batch_results = self.fetch_multiple_stocks_concurrent(batch, period, interval)
                    results.update(batch_results)
        
        # Record performance metrics
        fetch_time = time.time() - start_time
        total_data_points = sum(len(df) for df in results.values() if not df.empty)
        
        performance_monitor.record_fetch(
            symbols=symbols,
            fetch_time=fetch_time,
            cache_hits=len(cached_symbols),
            cache_misses=len(remaining_symbols),
            method_used="smart_fetch",
            data_points=total_data_points
        )
        
        logger.info(f"Smart fetch completed: {len(results)}/{len(symbols)} symbols retrieved in {fetch_time:.2f}s")
        return results
    
    def get_price_data(self, 
                      symbols: List[str], 
                      column: str = 'Close',
                      period: str = "2y", 
                      interval: str = "1d",
                      use_smart_fetch: bool = True) -> pd.DataFrame:
        """Get specific price column for multiple stocks as a DataFrame
        
        Args:
            symbols: List of stock symbols
            column: Price column to extract ('Open', 'High', 'Low', 'Close', 'Volume')
            period: Data period
            interval: Data interval
            use_smart_fetch: Whether to use optimized smart fetching (recommended)
        
        Returns:
            DataFrame with symbols as columns and dates as index
        """
        # Use optimized fetching method
        if use_smart_fetch:
            stock_data = self.smart_fetch_multiple_stocks(symbols, period, interval)
        else:
            stock_data = self.fetch_multiple_stocks(symbols, period, interval)
        
        price_data = pd.DataFrame()
        failed_symbols = []
        successful_symbols = []
        
        for symbol, data in stock_data.items():
            if not data.empty and column in data.columns:
                price_data[symbol] = data[column]
                successful_symbols.append(symbol)
            else:
                failed_symbols.append(symbol)
        
        # Log information about failed symbols
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        if successful_symbols:
            logger.info(f"Successfully fetched data for {len(successful_symbols)} symbols: {successful_symbols}")
        
        # Apply dropna and check if we have any data left
        clean_data = price_data.dropna()
        
        if clean_data.empty and not price_data.empty:
            logger.warning("All data was dropped due to missing values. This might indicate data quality issues.")
        
        return clean_data
    
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