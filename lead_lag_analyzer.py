import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, bootstrap
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class LeadLagResult:
    """Container for lead-lag analysis results"""
    leader_symbol: str
    follower_symbol: str
    lag_days: int
    correlation: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    method: str
    adjusted_p_value: Optional[float] = None
    bootstrap_ci: Optional[Tuple[float, float]] = None
    monte_carlo_p_value: Optional[float] = None
    rolling_stability: Optional[float] = None
    regime_consistency: Optional[Dict[str, float]] = None
    
    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        p_val = self.adjusted_p_value if self.adjusted_p_value is not None else self.p_value
        return p_val < alpha
    
    @property
    def abs_correlation(self) -> float:
        """Absolute value of correlation"""
        return abs(self.correlation)
    
    @property
    def strength_category(self) -> str:
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.7:
            return "Very Strong"
        elif abs_corr >= 0.5:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        elif abs_corr >= 0.1:
            return "Weak"
        else:
            return "Very Weak"
    
    @property
    def confidence_score(self) -> float:
        """Calculate overall confidence score based on multiple factors"""
        score = 0.0
        
        # Correlation strength (0-40 points)
        score += min(self.abs_correlation * 40, 40)
        
        # Statistical significance (0-30 points)
        p_val = self.adjusted_p_value if self.adjusted_p_value is not None else self.p_value
        if p_val < 0.001:
            score += 30
        elif p_val < 0.01:
            score += 25
        elif p_val < 0.05:
            score += 20
        elif p_val < 0.1:
            score += 10
        
        # Rolling stability (0-20 points)
        if self.rolling_stability is not None:
            score += self.rolling_stability * 20
        
        # Sample size bonus (0-10 points)
        if self.sample_size > 1000:
            score += 10
        elif self.sample_size > 500:
            score += 7
        elif self.sample_size > 250:
            score += 5
        elif self.sample_size > 100:
            score += 3
        
        return min(score, 100)  # Cap at 100

class LeadLagAnalyzer:
    """Analyzes lead-lag relationships between stock price series with advanced statistical methods"""
    
    def __init__(self, min_periods: int = 30, enable_monte_carlo: bool = True, 
                 enable_bootstrap: bool = True, enable_rolling_analysis: bool = True,
                 monte_carlo_iterations: int = 1000, bootstrap_iterations: int = 1000,
                 rolling_window_size: int = 252):
        """
        Initialize the analyzer with advanced statistical options
        
        Args:
            min_periods: Minimum number of observations required for analysis
            enable_monte_carlo: Whether to perform Monte Carlo significance testing
            enable_bootstrap: Whether to calculate bootstrap confidence intervals
            enable_rolling_analysis: Whether to perform rolling window stability analysis
            monte_carlo_iterations: Number of Monte Carlo simulations
            bootstrap_iterations: Number of bootstrap samples
            rolling_window_size: Size of rolling window for stability analysis (default: 252 trading days)
        """
        self.min_periods = min_periods
        self.enable_monte_carlo = enable_monte_carlo
        self.enable_bootstrap = enable_bootstrap
        self.enable_rolling_analysis = enable_rolling_analysis
        self.monte_carlo_iterations = monte_carlo_iterations
        self.bootstrap_iterations = bootstrap_iterations
        self.rolling_window_size = rolling_window_size
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'pct_change') -> pd.DataFrame:
        """Calculate returns from price data
        
        Args:
            prices: DataFrame with price data
            method: 'pct_change', 'log_returns', or 'diff'
        
        Returns:
            DataFrame with returns
        """
        if method == 'pct_change':
            returns = prices.pct_change()
        elif method == 'log_returns':
            returns = np.log(prices / prices.shift(1))
        elif method == 'diff':
            returns = prices.diff()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns.dropna()
    
    def create_lagged_series(self, series: pd.Series, lag: int) -> pd.Series:
        """Create a lagged version of the series
        
        Args:
            series: Time series data
            lag: Number of periods to lag (positive for backward lag)
        
        Returns:
            Lagged series
        """
        if lag > 0:
            return series.shift(lag)
        elif lag < 0:
            return series.shift(lag)
        else:
            return series.copy()
    
    def calculate_correlation_with_ci(self, 
                                    x: pd.Series, 
                                    y: pd.Series, 
                                    method: str = 'pearson',
                                    confidence_level: float = 0.95) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate correlation with confidence interval
        
        Args:
            x, y: Series to correlate
            method: 'pearson' or 'spearman'
            confidence_level: Confidence level for interval
        
        Returns:
            Tuple of (correlation, p_value, confidence_interval)
        """
        # Remove NaN values
        valid_data = pd.concat([x, y], axis=1).dropna()
        if len(valid_data) < self.min_periods:
            return np.nan, np.nan, (np.nan, np.nan)
        
        x_clean = valid_data.iloc[:, 0]
        y_clean = valid_data.iloc[:, 1]
        
        # Calculate correlation
        if method == 'pearson':
            corr, p_value = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_value = spearmanr(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate confidence interval using Fisher transformation
        n = len(x_clean)
        if method == 'pearson' and n > 3:
            # Fisher z-transformation
            z = np.arctanh(corr)
            se = 1 / np.sqrt(n - 3)
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            
            # Transform back
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
            confidence_interval = (ci_lower, ci_upper)
        else:
            # For Spearman or small samples, use bootstrap or simple approximation
            se_approx = np.sqrt((1 - corr**2) / (n - 2))
            margin = stats.t.ppf(1 - (1-confidence_level)/2, n-2) * se_approx
            confidence_interval = (max(-1, corr - margin), min(1, corr + margin))
        
        return corr, p_value, confidence_interval
    
    def monte_carlo_significance_test(self, x: pd.Series, y: pd.Series, 
                                    observed_correlation: float, 
                                    method: str = 'pearson',
                                    n_iterations: int = 1000) -> float:
        """Perform Monte Carlo significance test for correlation
        
        Args:
            x, y: Original series
            observed_correlation: The observed correlation to test
            method: Correlation method
            n_iterations: Number of Monte Carlo iterations
        
        Returns:
            Monte Carlo p-value
        """
        if not self.enable_monte_carlo:
            return np.nan
            
        # Remove NaN values
        valid_data = pd.concat([x, y], axis=1).dropna()
        if len(valid_data) < self.min_periods:
            return np.nan
            
        x_clean = valid_data.iloc[:, 0].values
        y_clean = valid_data.iloc[:, 1].values
        
        # Generate null distribution by shuffling one series
        null_correlations = []
        for _ in range(n_iterations):
            y_shuffled = np.random.permutation(y_clean)
            
            if method == 'pearson':
                null_corr, _ = pearsonr(x_clean, y_shuffled)
            elif method == 'spearman':
                null_corr, _ = spearmanr(x_clean, y_shuffled)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            null_correlations.append(null_corr)
        
        null_correlations = np.array(null_correlations)
        
        # Calculate two-tailed p-value
        extreme_count = np.sum(np.abs(null_correlations) >= np.abs(observed_correlation))
        p_value = extreme_count / n_iterations
        
        return p_value
    
    def bootstrap_confidence_interval(self, x: pd.Series, y: pd.Series,
                                    method: str = 'pearson',
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for correlation
        
        Args:
            x, y: Series to correlate
            method: Correlation method
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Bootstrap confidence interval
        """
        if not self.enable_bootstrap:
            return (np.nan, np.nan)
            
        # Remove NaN values
        valid_data = pd.concat([x, y], axis=1).dropna()
        if len(valid_data) < self.min_periods:
            return (np.nan, np.nan)
            
        x_clean = valid_data.iloc[:, 0].values
        y_clean = valid_data.iloc[:, 1].values
        n = len(x_clean)
        
        # Bootstrap sampling
        bootstrap_correlations = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x_clean[indices]
            y_boot = y_clean[indices]
            
            if method == 'pearson':
                boot_corr, _ = pearsonr(x_boot, y_boot)
            elif method == 'spearman':
                boot_corr, _ = spearmanr(x_boot, y_boot)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            bootstrap_correlations.append(boot_corr)
        
        bootstrap_correlations = np.array(bootstrap_correlations)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_correlations, lower_percentile)
        ci_upper = np.percentile(bootstrap_correlations, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def calculate_rolling_stability(self, x: pd.Series, y: pd.Series,
                                  method: str = 'pearson',
                                  window_size: int = None) -> float:
        """Calculate rolling correlation stability
        
        Args:
            x, y: Series to analyze
            method: Correlation method
            window_size: Rolling window size
        
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if not self.enable_rolling_analysis:
            return np.nan
            
        if window_size is None:
            window_size = self.rolling_window_size
            
        # Remove NaN values
        valid_data = pd.concat([x, y], axis=1).dropna()
        if len(valid_data) < window_size + self.min_periods:
            return np.nan
            
        x_clean = valid_data.iloc[:, 0]
        y_clean = valid_data.iloc[:, 1]
        
        # Calculate rolling correlations
        rolling_correlations = []
        for i in range(window_size, len(valid_data)):
            x_window = x_clean.iloc[i-window_size:i]
            y_window = y_clean.iloc[i-window_size:i]
            
            if method == 'pearson':
                corr, _ = pearsonr(x_window, y_window)
            elif method == 'spearman':
                corr, _ = spearmanr(x_window, y_window)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            rolling_correlations.append(corr)
        
        if len(rolling_correlations) < 2:
            return np.nan
            
        rolling_correlations = np.array(rolling_correlations)
        
        # Calculate stability as 1 - coefficient of variation
        mean_corr = np.mean(rolling_correlations)
        std_corr = np.std(rolling_correlations)
        
        if mean_corr == 0:
            return 0.0
            
        cv = abs(std_corr / mean_corr)
        stability = max(0, 1 - cv)  # Higher stability = lower coefficient of variation
        
        return min(stability, 1.0)  # Cap at 1.0
    
    def analyze_lead_lag_pair(self, 
                             leader_data: pd.Series, 
                             follower_data: pd.Series,
                             leader_symbol: str,
                             follower_symbol: str,
                             max_lag: int = 10,
                             method: str = 'pearson',
                             return_method: str = 'pct_change') -> List[LeadLagResult]:
        """Analyze lead-lag relationship between two stocks
        
        Args:
            leader_data: Price data for potential leader
            follower_data: Price data for potential follower
            leader_symbol: Symbol of leader stock
            follower_symbol: Symbol of follower stock
            max_lag: Maximum lag to test
            method: Correlation method
            return_method: Method to calculate returns
        
        Returns:
            List of LeadLagResult objects
        """
        # Calculate returns
        leader_returns = self.calculate_returns(pd.DataFrame({'price': leader_data}), return_method)['price']
        follower_returns = self.calculate_returns(pd.DataFrame({'price': follower_data}), return_method)['price']
        
        results = []
        
        # Test different lags
        for lag in range(0, max_lag + 1):
            # Lag the leader (positive lag means leader leads by lag periods)
            lagged_leader = self.create_lagged_series(leader_returns, lag)
            
            # Calculate correlation
            corr, p_value, ci = self.calculate_correlation_with_ci(
                lagged_leader, follower_returns, method
            )
            
            if not np.isnan(corr):
                # Count valid observations
                valid_data = pd.concat([lagged_leader, follower_returns], axis=1).dropna()
                sample_size = len(valid_data)
                
                # Advanced statistical analysis
                monte_carlo_p = None
                bootstrap_ci = None
                rolling_stability = None
                
                if sample_size >= self.min_periods:
                    # Monte Carlo significance test
                    if self.enable_monte_carlo:
                        try:
                            monte_carlo_p = self.monte_carlo_significance_test(
                                lagged_leader, follower_returns, corr, method, 
                                self.monte_carlo_iterations
                            )
                        except Exception as e:
                            self.logger.warning(f"Monte Carlo test failed: {e}")
                    
                    # Bootstrap confidence interval
                    if self.enable_bootstrap:
                        try:
                            bootstrap_ci = self.bootstrap_confidence_interval(
                                lagged_leader, follower_returns, method, 0.95, 
                                self.bootstrap_iterations
                            )
                        except Exception as e:
                            self.logger.warning(f"Bootstrap CI failed: {e}")
                    
                    # Rolling stability analysis
                    if self.enable_rolling_analysis:
                        try:
                            rolling_stability = self.calculate_rolling_stability(
                                lagged_leader, follower_returns, method
                            )
                        except Exception as e:
                            self.logger.warning(f"Rolling stability analysis failed: {e}")
                
                result = LeadLagResult(
                    leader_symbol=leader_symbol,
                    follower_symbol=follower_symbol,
                    lag_days=lag,
                    correlation=corr,
                    p_value=p_value,
                    confidence_interval=ci,
                    sample_size=sample_size,
                    method=method,
                    bootstrap_ci=bootstrap_ci,
                    monte_carlo_p_value=monte_carlo_p,
                    rolling_stability=rolling_stability
                )
                results.append(result)
        
        return results
    
    def analyze_multiple_pairs(self, 
                              price_data: pd.DataFrame,
                              max_lag: int = 10,
                              method: str = 'pearson',
                              return_method: str = 'pct_change',
                              min_correlation: float = 0.1) -> pd.DataFrame:
        """Analyze lead-lag relationships for all stock pairs
        
        Args:
            price_data: DataFrame with stock prices (columns = symbols)
            max_lag: Maximum lag to test
            method: Correlation method
            return_method: Method to calculate returns
            min_correlation: Minimum absolute correlation to include
        
        Returns:
            DataFrame with all significant relationships
        """
        symbols = price_data.columns.tolist()
        all_results = []
        
        logger.info(f"Analyzing {len(symbols)} stocks with max lag of {max_lag} days")
        
        # Analyze all pairs
        for i, leader in enumerate(symbols):
            for j, follower in enumerate(symbols):
                if i != j:  # Don't analyze stock with itself
                    try:
                        results = self.analyze_lead_lag_pair(
                            price_data[leader],
                            price_data[follower],
                            leader,
                            follower,
                            max_lag,
                            method,
                            return_method
                        )
                        all_results.extend(results)
                    except Exception as e:
                        logger.warning(f"Failed to analyze {leader} -> {follower}: {e}")
        
        # Convert to DataFrame
        if not all_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame([
            {
                'leader': r.leader_symbol,
                'follower': r.follower_symbol,
                'lag_days': r.lag_days,
                'correlation': r.correlation,
                'abs_correlation': abs(r.correlation),
                'p_value': r.p_value,
                'ci_lower': r.confidence_interval[0],
                'ci_upper': r.confidence_interval[1],
                'sample_size': r.sample_size,
                'is_significant': r.is_significant,
                'strength': r.strength_category,
                'method': r.method
            }
            for r in all_results
        ])
        
        # Apply multiple comparison correction
        if len(results_df) > 1:
            try:
                # Benjamini-Hochberg FDR correction
                rejected, adjusted_p_values, _, _ = multipletests(
                    results_df['p_value'].values, 
                    alpha=0.05, 
                    method='fdr_bh'
                )
                results_df['adjusted_p_value'] = adjusted_p_values
                results_df['is_significant_adjusted'] = rejected
            except Exception as e:
                self.logger.warning(f"Multiple comparison correction failed: {e}")
                results_df['adjusted_p_value'] = results_df['p_value']
                results_df['is_significant_adjusted'] = results_df['is_significant']
        else:
            results_df['adjusted_p_value'] = results_df['p_value']
            results_df['is_significant_adjusted'] = results_df['is_significant']
        
        # Add advanced metrics to DataFrame
        for i, result in enumerate(all_results):
            if i < len(results_df):
                if hasattr(result, 'monte_carlo_p_value') and result.monte_carlo_p_value is not None:
                    results_df.loc[i, 'monte_carlo_p_value'] = result.monte_carlo_p_value
                if hasattr(result, 'bootstrap_ci') and result.bootstrap_ci is not None:
                    results_df.loc[i, 'bootstrap_ci_lower'] = result.bootstrap_ci[0]
                    results_df.loc[i, 'bootstrap_ci_upper'] = result.bootstrap_ci[1]
                if hasattr(result, 'rolling_stability') and result.rolling_stability is not None:
                    results_df.loc[i, 'rolling_stability'] = result.rolling_stability
                
                # Calculate confidence score
                results_df.loc[i, 'confidence_score'] = result.confidence_score
        
        # Filter by minimum correlation
        results_df = results_df[results_df['abs_correlation'] >= min_correlation]
        
        # Sort by confidence score descending, then by absolute correlation
        if 'confidence_score' in results_df.columns:
            results_df = results_df.sort_values(['confidence_score', 'abs_correlation'], ascending=[False, False])
        else:
            results_df = results_df.sort_values('abs_correlation', ascending=False)
        
        return results_df.reset_index(drop=True)
    
    def find_highest_correlations(self,
                                 price_data: pd.DataFrame,
                                 top_n: int = 10,
                                 method: str = 'pearson',
                                 return_method: str = 'pct_change',
                                 min_correlation: float = 0.5,
                                 lag_days: int = 0) -> pd.DataFrame:
        """Find stocks with highest correlations in a given universe
        
        Args:
            price_data: DataFrame with stock prices (columns = symbols)
            top_n: Number of top correlated pairs to return
            method: Correlation method ('pearson' or 'spearman')
            return_method: Method to calculate returns
            min_correlation: Minimum absolute correlation threshold
            lag_days: Specific lag to analyze (0 for contemporaneous)
        
        Returns:
            DataFrame with highest correlated stock pairs
        """
        symbols = price_data.columns.tolist()
        correlation_results = []
        
        logger.info(f"Finding highest correlations among {len(symbols)} stocks")
        
        # Calculate returns once
        returns_data = self.calculate_returns(price_data, return_method)
        
        # Analyze all unique pairs (avoid duplicates like A-B and B-A)
        for i, stock1 in enumerate(symbols):
            for j, stock2 in enumerate(symbols[i+1:], i+1):
                try:
                    # Get return series
                    returns1 = returns_data[stock1]
                    returns2 = returns_data[stock2]
                    
                    # Apply lag if specified
                    if lag_days > 0:
                        returns1 = self.create_lagged_series(returns1, lag_days)
                    elif lag_days < 0:
                        returns2 = self.create_lagged_series(returns2, abs(lag_days))
                    
                    # Calculate correlation with confidence interval
                    corr, p_value, ci = self.calculate_correlation_with_ci(
                        returns1, returns2, method
                    )
                    
                    if not np.isnan(corr) and abs(corr) >= min_correlation:
                        # Calculate additional statistics if enabled
                        bootstrap_ci = None
                        monte_carlo_p = None
                        
                        if self.enable_bootstrap:
                            bootstrap_ci = self.bootstrap_confidence_interval(
                                returns1, returns2, method
                            )
                        
                        if self.enable_monte_carlo:
                            monte_carlo_p = self.monte_carlo_significance_test(
                                returns1, returns2, corr, method
                            )
                        
                        # Determine sample size
                        valid_data = pd.concat([returns1, returns2], axis=1).dropna()
                        sample_size = len(valid_data)
                        
                        correlation_results.append({
                            'stock1': stock1,
                            'stock2': stock2,
                            'correlation': corr,
                            'abs_correlation': abs(corr),
                            'p_value': p_value,
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'bootstrap_ci_lower': bootstrap_ci[0] if bootstrap_ci else np.nan,
                            'bootstrap_ci_upper': bootstrap_ci[1] if bootstrap_ci else np.nan,
                            'monte_carlo_p_value': monte_carlo_p,
                            'sample_size': sample_size,
                            'lag_days': lag_days,
                            'method': method,
                            'is_significant': p_value < 0.05 if not np.isnan(p_value) else False,
                            'strength': self._get_correlation_strength(abs(corr))
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze correlation {stock1}-{stock2}: {e}")
        
        if not correlation_results:
            logger.warning(f"No correlations found above threshold {min_correlation}")
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by absolute correlation
        results_df = pd.DataFrame(correlation_results)
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        
        # Apply multiple comparison correction if we have multiple results
        if len(results_df) > 1:
            try:
                rejected, adjusted_p_values, _, _ = multipletests(
                    results_df['p_value'].values,
                    alpha=0.05,
                    method='fdr_bh'
                )
                results_df['adjusted_p_value'] = adjusted_p_values
                results_df['is_significant_adjusted'] = rejected
            except Exception as e:
                logger.warning(f"Multiple comparison correction failed: {e}")
                results_df['adjusted_p_value'] = results_df['p_value']
                results_df['is_significant_adjusted'] = results_df['is_significant']
        
        # Return top N results
        return results_df.head(top_n)
    
    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Categorize correlation strength"""
        if abs_corr >= 0.8:
            return 'Very Strong'
        elif abs_corr >= 0.6:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        elif abs_corr >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def find_best_predictors(self, 
                           results_df: pd.DataFrame, 
                           target_stock: str,
                           top_n: int = 10,
                           min_significance: float = 0.05) -> pd.DataFrame:
        """Find the best leading indicators for a target stock
        
        Args:
            results_df: Results from analyze_multiple_pairs
            target_stock: Stock symbol to find predictors for
            top_n: Number of top predictors to return
            min_significance: Maximum p-value for significance
        
        Returns:
            DataFrame with top predictors
        """
        # Filter for target stock as follower and significant results
        predictors = results_df[
            (results_df['follower'] == target_stock) & 
            (results_df['p_value'] <= min_significance) &
            (results_df['lag_days'] > 0)  # Only leading relationships
        ].copy()
        
        if predictors.empty:
            logger.warning(f"No significant predictors found for {target_stock}")
            return pd.DataFrame()
        
        # Sort by absolute correlation and take top N
        top_predictors = predictors.nlargest(top_n, 'abs_correlation')
        
        return top_predictors
    
    def create_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """Create a summary report of the analysis
        
        Args:
            results_df: Results from analyze_multiple_pairs
        
        Returns:
            Dictionary with summary statistics
        """
        if results_df.empty:
            return {"message": "No results to summarize"}
        
        summary = {
            'total_relationships': len(results_df),
            'significant_relationships': len(results_df[results_df['is_significant']]),
            'avg_correlation': results_df['correlation'].mean(),
            'avg_abs_correlation': results_df['abs_correlation'].mean(),
            'max_correlation': results_df['abs_correlation'].max(),
            'correlation_distribution': results_df['strength'].value_counts().to_dict(),
            'lag_distribution': results_df['lag_days'].value_counts().sort_index().to_dict(),
        }
        
        # Add advanced statistics if available
        if 'adjusted_p_value' in results_df.columns:
            summary['significant_relationships_adjusted'] = len(results_df[results_df['is_significant_adjusted']])
            summary['avg_adjusted_p_value'] = results_df['adjusted_p_value'].mean()
        
        if 'confidence_score' in results_df.columns:
            summary['avg_confidence_score'] = results_df['confidence_score'].mean()
            summary['high_confidence_relationships'] = len(results_df[results_df['confidence_score'] >= 70])
        
        if 'rolling_stability' in results_df.columns:
            stability_data = results_df['rolling_stability'].dropna()
            if len(stability_data) > 0:
                summary['avg_rolling_stability'] = stability_data.mean()
                summary['stable_relationships'] = len(stability_data[stability_data >= 0.7])
        
        # Most predictive pairs (use confidence score if available, otherwise correlation)
        sort_column = 'confidence_score' if 'confidence_score' in results_df.columns else 'abs_correlation'
        top_pairs_cols = ['leader', 'follower', 'lag_days', 'correlation', 'p_value']
        if 'confidence_score' in results_df.columns:
            top_pairs_cols.append('confidence_score')
        
        summary['most_predictive_pairs'] = results_df.nlargest(5, sort_column)[top_pairs_cols].to_dict('records')
        
        return summary
    
    def summarize_results(self, results_df: pd.DataFrame, apply_correction: bool = True) -> pd.DataFrame:
        """Apply post-processing to analysis results
        
        Args:
            results_df: Results DataFrame from analyze_multiple_pairs
            apply_correction: Whether to apply multiple comparison correction
        
        Returns:
            Processed DataFrame with corrections applied
        """
        if results_df.empty:
            return results_df
        
        results_copy = results_df.copy()
        
        # Apply multiple comparison correction if requested
        if apply_correction and len(results_copy) > 1:
            try:
                # Use adjusted p-values if they exist, otherwise use original p-values
                p_values = results_copy.get('adjusted_p_value', results_copy['p_value']).values
                
                # Benjamini-Hochberg FDR correction
                rejected, adjusted_p_values, _, _ = multipletests(
                    p_values, 
                    alpha=0.05, 
                    method='fdr_bh'
                )
                
                results_copy['adjusted_p_value'] = adjusted_p_values
                results_copy['is_significant_adjusted'] = rejected
                
                self.logger.info(f"Applied multiple comparison correction: {sum(rejected)}/{len(rejected)} relationships remain significant")
                
            except Exception as e:
                self.logger.warning(f"Multiple comparison correction failed: {e}")
                # Fallback: use original values
                if 'adjusted_p_value' not in results_copy.columns:
                    results_copy['adjusted_p_value'] = results_copy['p_value']
                if 'is_significant_adjusted' not in results_copy.columns:
                    results_copy['is_significant_adjusted'] = results_copy['is_significant']
        else:
            # No correction needed or requested
            if 'adjusted_p_value' not in results_copy.columns:
                results_copy['adjusted_p_value'] = results_copy['p_value']
            if 'is_significant_adjusted' not in results_copy.columns:
                results_copy['is_significant_adjusted'] = results_copy['is_significant']
        
        return results_copy

# Example usage
if __name__ == "__main__":
    # This would typically be called with real data
    analyzer = LeadLagAnalyzer(
        enable_monte_carlo=True,
        enable_bootstrap=True, 
        enable_rolling_analysis=True,
        monte_carlo_iterations=500,  # Reduced for demo
        bootstrap_iterations=500     # Reduced for demo
    )
    
    # Create sample data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate correlated stock data
    base_returns = np.random.normal(0.001, 0.02, len(dates))
    stock_a = 100 * np.exp(np.cumsum(base_returns))
    stock_b = 100 * np.exp(np.cumsum(base_returns + np.random.normal(0, 0.01, len(dates))))
    
    price_data = pd.DataFrame({
        'STOCK_A': stock_a,
        'STOCK_B': stock_b
    }, index=dates)
    
    # Analyze relationships
    results = analyzer.analyze_multiple_pairs(price_data, max_lag=5)
    print("Lead-Lag Analysis Results:")
    print(results.head())
    
    # Create summary
    summary = analyzer.create_summary_report(results)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")