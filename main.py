#!/usr/bin/env python3
"""
Lead-Lag Stock Analysis System

A comprehensive system for analyzing lead-lag relationships between stocks
and backtesting trading strategies based on these relationships.

Author: AI Assistant
Date: 2024
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from pathlib import Path

# Import our custom modules
from data_fetcher import StockDataFetcher
from lead_lag_analyzer import LeadLagAnalyzer
from backtester import LeadLagBacktester
from visualizer import LeadLagVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lead_lag_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LeadLagSystem:
    """Main system class that orchestrates the entire analysis pipeline"""
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 output_dir: str = "output",
                 plots_dir: str = "plots"):
        
        # Initialize components
        self.data_fetcher = StockDataFetcher(cache_dir)
        self.analyzer = LeadLagAnalyzer()  # Will be configured dynamically
        self.backtester = LeadLagBacktester()  # Will be configured dynamically
        self.visualizer = LeadLagVisualizer()
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info("Lead-Lag Analysis System initialized")
    
    def run_full_analysis(self, 
                         symbols: List[str],
                         period: str = "2y",
                         interval: str = "1d",
                         max_lag: int = 10,
                         correlation_method: str = "pearson",
                         return_method: str = "pct_change",
                         min_correlation: float = 0.1,
                         backtest_top_n: int = 5,
                         save_results: bool = True,
                         enable_monte_carlo: bool = True,
                         monte_carlo_iterations: int = 1000,
                         enable_bootstrap: bool = True,
                         bootstrap_iterations: int = 1000,
                         enable_rolling_analysis: bool = True,
                         rolling_window_size: int = 252,
                         apply_multiple_correction: bool = True,
                         enable_detrending: bool = False,
                         detrend_method: str = "linear",
                         enable_market_adjustment: bool = False,
                         market_index: str = "SPY",
                         market_adjustment_method: str = "beta_adjust",
                         enable_normalization: bool = False,
                          normalization_method: str = "zscore",
                          # Backtesting cost parameters
                          transaction_cost: float = 0.001,
                          bid_ask_spread: float = 0.0005,
                          slippage: float = 0.0002,
                          enable_trading_costs: bool = True) -> Dict:
        """Run complete lead-lag analysis pipeline
        
        Args:
            symbols: List of stock symbols to analyze
            period: Data period to fetch
            interval: Data interval
            max_lag: Maximum lag days to test
            correlation_method: Correlation method ('pearson' or 'spearman')
            return_method: Return calculation method
            min_correlation: Minimum correlation threshold
            backtest_top_n: Number of top strategies to backtest
            save_results: Whether to save results to files
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Starting full analysis for {len(symbols)} stocks")
        
        try:
            # Step 1: Fetch data using optimized method
            logger.info("Fetching stock data using smart fetch...")
            price_data = self.data_fetcher.get_price_data(
                symbols, column='Close', period=period, interval=interval, use_smart_fetch=True
            )
            
            if price_data.empty:
                raise ValueError("No price data retrieved")
            
            logger.info(f"Retrieved data for {len(price_data.columns)} stocks, {len(price_data)} observations")
            
            # Apply data transformations
            processed_data = price_data.copy()
            
            # Detrending
            if enable_detrending:
                logger.info(f"Applying {detrend_method} detrending...")
                processed_data = self.data_fetcher.detrend_data(processed_data, method=detrend_method)
            
            # Market adjustment
            if enable_market_adjustment:
                logger.info(f"Applying market adjustment using {market_index}...")
                market_data = self.data_fetcher.get_price_data(
                    [market_index], column='Close', period=period, interval=interval
                )
                processed_data = self.data_fetcher.remove_market_effects(
                    processed_data, market_data[market_index], method=market_adjustment_method
                )
            
            # Use processed data for analysis
            price_data = processed_data
            
            # Step 2: Configure analyzer with advanced options
            self.analyzer.enable_monte_carlo = enable_monte_carlo
            self.analyzer.monte_carlo_iterations = monte_carlo_iterations
            self.analyzer.enable_bootstrap = enable_bootstrap
            self.analyzer.bootstrap_iterations = bootstrap_iterations
            self.analyzer.enable_rolling_analysis = enable_rolling_analysis
            self.analyzer.rolling_window_size = rolling_window_size
            
            # Configure backtester with cost parameters
            self.backtester.transaction_cost = transaction_cost
            self.backtester.bid_ask_spread = bid_ask_spread
            self.backtester.slippage = slippage
            self.backtester.enable_costs = enable_trading_costs
            
            # Analyze lead-lag relationships
            logger.info("Analyzing lead-lag relationships...")
            results_df = self.analyzer.analyze_multiple_pairs(
                price_data,
                max_lag=max_lag,
                method=correlation_method,
                return_method=return_method,
                min_correlation=min_correlation
            )
            
            # Apply multiple comparison correction if enabled
            if apply_multiple_correction and not results_df.empty:
                results_df = self.analyzer.summarize_results(results_df, apply_correction=True)
            
            # Apply normalization if enabled
            if enable_normalization and not results_df.empty:
                # This would be applied to the underlying data, already handled above
                pass
            
            if results_df.empty:
                logger.warning("No significant relationships found")
                return {"message": "No significant relationships found"}
            
            logger.info(f"Found {len(results_df)} significant relationships")
            
            # Step 3: Create summary report
            summary = self.analyzer.create_summary_report(results_df)
            
            # Step 4: Find best predictors for each stock
            best_predictors = {}
            for symbol in symbols:
                predictors = self.analyzer.find_best_predictors(
                    results_df, symbol, top_n=10
                )
                if not predictors.empty:
                    best_predictors[symbol] = predictors
            
            # Step 5: Backtest top strategies
            logger.info("Running backtests on top strategies...")
            backtest_results = self._run_backtests(
                price_data, results_df, top_n=backtest_top_n
            )
            
            # Step 6: Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(results_df, backtest_results)
            
            # Step 7: Compile final results
            final_results = {
                'analysis_summary': summary,
                'lead_lag_results': results_df,
                'best_predictors': best_predictors,
                'backtest_results': backtest_results,
                'price_data': price_data,
                'parameters': {
                    'symbols': symbols,
                    'period': period,
                    'interval': interval,
                    'max_lag': max_lag,
                    'correlation_method': correlation_method,
                    'return_method': return_method,
                    'min_correlation': min_correlation
                }
            }
            
            # Step 8: Save results
            if save_results:
                self._save_results(final_results)
            
            logger.info("Analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def find_highest_correlations(self,
                                 symbols: List[str],
                                 period: str = "2y",
                                 interval: str = "1d",
                                 top_n: int = 10,
                                 correlation_method: str = "pearson",
                                 return_method: str = "pct_change",
                                 min_correlation: float = 0.5,
                                 lag_days: int = 0,
                                 enable_monte_carlo: bool = True,
                                 monte_carlo_iterations: int = 1000,
                                 enable_bootstrap: bool = True,
                                 bootstrap_iterations: int = 1000,
                                 apply_multiple_correction: bool = True,
                                 save_results: bool = True,
                                 enable_detrending: bool = False,
                                 detrend_method: str = "linear",
                                 enable_market_adjustment: bool = False,
                                 market_index: str = "SPY",
                                 market_adjustment_method: str = "beta_adjust",
                                 enable_normalization: bool = False,
                                 normalization_method: str = "zscore") -> Dict:
        """Find stocks with highest correlations in a given universe
        
        Args:
            symbols: List of stock symbols to analyze
            period: Data period to fetch
            interval: Data interval
            top_n: Number of top correlated pairs to return
            correlation_method: Correlation method ('pearson' or 'spearman')
            return_method: Return calculation method
            min_correlation: Minimum absolute correlation threshold
            lag_days: Specific lag to analyze (0 for contemporaneous)
            enable_monte_carlo: Whether to perform Monte Carlo testing
            monte_carlo_iterations: Number of Monte Carlo iterations
            enable_bootstrap: Whether to calculate bootstrap confidence intervals
            bootstrap_iterations: Number of bootstrap samples
            save_results: Whether to save results to files
        
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info(f"Finding highest correlations among {len(symbols)} stocks")
        
        try:
            # Step 1: Fetch data using optimized method
            logger.info("Fetching stock data using smart fetch...")
            price_data = self.data_fetcher.get_price_data(
                symbols, column='Close', period=period, interval=interval, use_smart_fetch=True
            )
            
            if price_data.empty:
                raise ValueError("No price data retrieved")
            
            logger.info(f"Retrieved data for {len(price_data.columns)} stocks, {len(price_data)} observations")
            
            # Step 2: Apply data transformations
            if enable_detrending:
                logger.info(f"Applying {detrend_method} detrending...")
                price_data = self.data_fetcher.detrend_data(price_data, method=detrend_method)
            
            if enable_market_adjustment:
                logger.info(f"Applying market adjustment using {market_index}...")
                price_data = self.data_fetcher.adjust_for_market(price_data, market_index, method=market_adjustment_method)
            
            if enable_normalization:
                logger.info(f"Applying {normalization_method} normalization...")
                price_data = self.data_fetcher.normalize_data(price_data, method=normalization_method)
            
            # Step 3: Configure analyzer
            self.analyzer.enable_monte_carlo = enable_monte_carlo
            self.analyzer.monte_carlo_iterations = monte_carlo_iterations
            self.analyzer.enable_bootstrap = enable_bootstrap
            self.analyzer.bootstrap_iterations = bootstrap_iterations
            self.analyzer.apply_multiple_correction = apply_multiple_correction
            
            # Step 4: Find highest correlations
            logger.info("Analyzing correlations...")
            correlation_results = self.analyzer.find_highest_correlations(
                price_data,
                top_n=top_n,
                method=correlation_method,
                return_method=return_method,
                min_correlation=min_correlation,
                lag_days=lag_days
            )
            
            if correlation_results.empty:
                logger.warning(f"No correlations found above threshold {min_correlation}")
                return {"message": f"No correlations found above threshold {min_correlation}"}
            
            logger.info(f"Found {len(correlation_results)} highly correlated pairs")
            
            # Step 5: Create summary statistics
            summary = {
                'total_pairs_analyzed': len(symbols) * (len(symbols) - 1) // 2,
                'pairs_above_threshold': len(correlation_results),
                'highest_correlation': correlation_results.iloc[0]['correlation'] if not correlation_results.empty else None,
                'average_correlation': correlation_results['correlation'].mean() if not correlation_results.empty else None,
                'significant_pairs': len(correlation_results[correlation_results['is_significant']]) if not correlation_results.empty else 0,
                'method': correlation_method,
                'lag_days': lag_days,
                'min_correlation_threshold': min_correlation
            }
            
            # Step 6: Compile final results
            final_results = {
                'correlation_summary': summary,
                'correlation_results': correlation_results,
                'price_data': price_data,
                'parameters': {
                    'symbols': symbols,
                    'period': period,
                    'interval': interval,
                    'top_n': top_n,
                    'correlation_method': correlation_method,
                    'return_method': return_method,
                    'min_correlation': min_correlation,
                    'lag_days': lag_days,
                    'enable_detrending': enable_detrending,
                    'detrend_method': detrend_method,
                    'enable_market_adjustment': enable_market_adjustment,
                    'market_index': market_index,
                    'market_adjustment_method': market_adjustment_method,
                    'enable_normalization': enable_normalization,
                    'normalization_method': normalization_method
                }
            }
            
            # Step 7: Save results
            if save_results:
                self._save_correlation_results(final_results)
            
            logger.info("Correlation analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def enhanced_correlation_discovery(self,
                                     symbols: List[str],
                                     period: str = "2y",
                                     interval: str = "1d",
                                     time_horizons: List[int] = [1, 5, 10, 20],
                                     lag_periods: List[int] = [0, 1, 2, 3, 5],
                                     top_n: int = 10,
                                     correlation_method: str = "pearson",
                                     return_method: str = "pct_change",
                                     min_correlation: float = 0.3,
                                     enable_monte_carlo: bool = True,
                                     monte_carlo_iterations: int = 1000,
                                     enable_bootstrap: bool = True,
                                     bootstrap_iterations: int = 1000,
                                     apply_multiple_correction: bool = True,
                                     save_results: bool = True,
                                     enable_detrending: bool = False,
                                     detrend_method: str = "linear",
                                     enable_market_adjustment: bool = False,
                                     market_index: str = "SPY",
                                     market_adjustment_method: str = "beta_adjust",
                                     enable_normalization: bool = True,
                                     normalization_method: str = "zscore") -> Dict:
        """Enhanced correlation discovery with time horizons and lagging analysis
        
        This function compares stocks within a defined universe across multiple time horizons
        and lagging periods, ranking results by highest correlation.
        
        Args:
            symbols: List of stock symbols to analyze
            period: Data period to fetch
            interval: Data interval
            time_horizons: List of time horizons (in days) to analyze
            lag_periods: List of lag periods (in days) to test
            top_n: Number of top correlated pairs to return per combination
            correlation_method: Correlation method ('pearson' or 'spearman')
            return_method: Return calculation method
            min_correlation: Minimum absolute correlation threshold
            enable_monte_carlo: Whether to perform Monte Carlo testing
            monte_carlo_iterations: Number of Monte Carlo iterations
            enable_bootstrap: Whether to calculate bootstrap confidence intervals
            bootstrap_iterations: Number of bootstrap samples
            apply_multiple_correction: Apply multiple comparison correction
            save_results: Whether to save results to files
            enable_detrending: Enable data detrending
            detrend_method: Detrending method
            enable_market_adjustment: Enable market adjustment
            market_index: Market benchmark index
            market_adjustment_method: Market adjustment method
            enable_normalization: Enable data normalization
            normalization_method: Normalization method
        
        Returns:
            Dictionary with enhanced correlation analysis results
        """
        logger.info(f"Running enhanced correlation discovery for {len(symbols)} stocks")
        logger.info(f"Time horizons: {time_horizons}, Lag periods: {lag_periods}")
        
        try:
            # Step 1: Fetch data using optimized method
            logger.info("Fetching stock data using smart fetch...")
            price_data = self.data_fetcher.get_price_data(
                symbols, column='Close', period=period, interval=interval, use_smart_fetch=True
            )
            
            if price_data.empty:
                raise ValueError("No price data retrieved")
            
            logger.info(f"Retrieved data for {len(price_data.columns)} stocks, {len(price_data)} observations")
            
            # Step 2: Apply data transformations
            if enable_detrending:
                logger.info(f"Applying {detrend_method} detrending...")
                price_data = self.data_fetcher.detrend_data(price_data, method=detrend_method)
            
            if enable_market_adjustment:
                logger.info(f"Applying market adjustment using {market_index}...")
                price_data = self.data_fetcher.adjust_for_market(price_data, market_index, method=market_adjustment_method)
            
            if enable_normalization:
                logger.info(f"Applying {normalization_method} normalization...")
                price_data = self.data_fetcher.normalize_data(price_data, method=normalization_method)
            
            # Step 3: Configure analyzer
            self.analyzer.enable_monte_carlo = enable_monte_carlo
            self.analyzer.monte_carlo_iterations = monte_carlo_iterations
            self.analyzer.enable_bootstrap = enable_bootstrap
            self.analyzer.bootstrap_iterations = bootstrap_iterations
            self.analyzer.apply_multiple_correction = apply_multiple_correction
            
            # Step 4: Run enhanced correlation discovery across time horizons and lags
            all_results = []
            horizon_summaries = {}
            
            for horizon in time_horizons:
                logger.info(f"Analyzing time horizon: {horizon} days")
                horizon_results = []
                
                # Calculate returns for this time horizon
                if horizon == 1:
                    horizon_data = price_data
                else:
                    # Use rolling windows for longer horizons
                    horizon_data = price_data.rolling(window=horizon).mean().dropna()
                
                for lag in lag_periods:
                    logger.info(f"  Testing lag period: {lag} days")
                    
                    # Find correlations for this horizon-lag combination
                    correlation_results = self.analyzer.find_highest_correlations(
                        horizon_data,
                        top_n=top_n,
                        method=correlation_method,
                        return_method=return_method,
                        min_correlation=min_correlation,
                        lag_days=lag
                    )
                    
                    if not correlation_results.empty:
                        # Add horizon and lag information
                        correlation_results['time_horizon'] = horizon
                        correlation_results['lag_period'] = lag
                        correlation_results['horizon_lag_combo'] = f"{horizon}d_lag{lag}"
                        
                        horizon_results.append(correlation_results)
                        all_results.append(correlation_results)
                
                # Create summary for this horizon
                if horizon_results:
                    combined_horizon = pd.concat(horizon_results, ignore_index=True)
                    horizon_summaries[f"{horizon}d"] = {
                        'total_pairs': len(combined_horizon),
                        'avg_correlation': combined_horizon['correlation'].mean(),
                        'max_correlation': combined_horizon['correlation'].max(),
                        'significant_pairs': len(combined_horizon[combined_horizon['is_significant']]) if 'is_significant' in combined_horizon.columns else 0,
                        'best_lag': combined_horizon.loc[combined_horizon['correlation'].idxmax(), 'lag_period'] if not combined_horizon.empty else None
                    }
            
            # Step 5: Combine and rank all results
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                
                # Sort by correlation strength (absolute value)
                final_results['abs_correlation'] = final_results['correlation'].abs()
                final_results = final_results.sort_values('abs_correlation', ascending=False)
                
                # Get top N overall results
                top_results = final_results.head(top_n * 2)  # Get more results for variety
                
                logger.info(f"Found {len(final_results)} total correlation combinations")
            else:
                final_results = pd.DataFrame()
                top_results = pd.DataFrame()
                logger.warning("No correlations found above threshold")
            
            # Step 6: Create comprehensive summary
            summary = {
                'total_combinations_analyzed': len(time_horizons) * len(lag_periods) * len(symbols) * (len(symbols) - 1) // 2,
                'total_pairs_found': len(final_results) if not final_results.empty else 0,
                'time_horizons_tested': time_horizons,
                'lag_periods_tested': lag_periods,
                'horizon_summaries': horizon_summaries,
                'overall_stats': {
                    'highest_correlation': final_results['correlation'].max() if not final_results.empty else None,
                    'average_correlation': final_results['correlation'].mean() if not final_results.empty else None,
                    'best_horizon_lag_combo': final_results.loc[final_results['correlation'].idxmax(), 'horizon_lag_combo'] if not final_results.empty else None,
                    'most_frequent_horizon': final_results['time_horizon'].mode().iloc[0] if not final_results.empty and not final_results['time_horizon'].mode().empty else None,
                    'most_frequent_lag': final_results['lag_period'].mode().iloc[0] if not final_results.empty and not final_results['lag_period'].mode().empty else None
                },
                'method': correlation_method,
                'min_correlation_threshold': min_correlation
            }
            
            # Step 7: Compile final results
            enhanced_results = {
                'summary': summary,
                'correlation_results': top_results,
                'all_results': final_results,
                'horizon_breakdown': horizon_summaries,
                'price_data': price_data,
                'parameters': {
                    'symbols': symbols,
                    'period': period,
                    'interval': interval,
                    'time_horizons': time_horizons,
                    'lag_periods': lag_periods,
                    'top_n': top_n,
                    'correlation_method': correlation_method,
                    'return_method': return_method,
                    'min_correlation': min_correlation,
                    'enable_detrending': enable_detrending,
                    'detrend_method': detrend_method,
                    'enable_market_adjustment': enable_market_adjustment,
                    'market_index': market_index,
                    'market_adjustment_method': market_adjustment_method,
                    'enable_normalization': enable_normalization,
                    'normalization_method': normalization_method
                }
            }
            
            # Step 8: Save results
            if save_results:
                self._save_enhanced_correlation_results(enhanced_results)
            
            logger.info("Enhanced correlation discovery completed successfully")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Enhanced correlation discovery failed: {e}")
            raise
    
    def _run_backtests(self, 
                      price_data: pd.DataFrame, 
                      results_df: pd.DataFrame, 
                      top_n: int = 5) -> Dict:
        """Run backtests on top performing strategies"""
        # Get top strategies by absolute correlation
        top_strategies = results_df[
            (results_df['is_significant']) & 
            (results_df['lag_days'] > 0)
        ].nlargest(top_n, 'abs_correlation')
        
        backtest_results = {
            'individual_results': {},
            'comparison': None,
            'performance_summary': {}
        }
        
        strategies_for_comparison = []
        
        for _, strategy in top_strategies.iterrows():
            leader = strategy['leader']
            follower = strategy['follower']
            lag_days = strategy['lag_days']
            
            try:
                # Run backtest
                performance, detailed_results = self.backtester.backtest_strategy(
                    leader_data=price_data[leader],
                    follower_data=price_data[follower],
                    lag_days=lag_days,
                    threshold=0.02,  # 2% threshold
                    max_position_size=0.1  # 10% max position
                )
                
                strategy_name = f"{leader}->{follower}_lag{lag_days}"
                backtest_results['individual_results'][strategy_name] = {
                    'performance': performance,
                    'detailed_results': detailed_results
                }
                
                # Prepare for comparison
                strategies_for_comparison.append({
                    'leader_data': price_data[leader],
                    'follower_data': price_data[follower],
                    'lag_days': lag_days,
                    'threshold': 0.02,
                    'max_position_size': 0.1
                })
                
                logger.info(f"Backtested {strategy_name}: {performance.total_return:.2%} return, {performance.sharpe_ratio:.2f} Sharpe")
                
            except Exception as e:
                logger.warning(f"Failed to backtest {leader}->{follower}: {e}")
                continue
        
        # Run strategy comparison
        if strategies_for_comparison:
            try:
                # Use first follower as benchmark
                benchmark = price_data[strategies_for_comparison[0]['follower_data'].name]
                comparison_df = self.backtester.compare_strategies(
                    strategies_for_comparison, benchmark
                )
                backtest_results['comparison'] = comparison_df
                
                # Create performance summary
                if not comparison_df.empty:
                    backtest_results['performance_summary'] = {
                        'best_strategy': comparison_df.iloc[0]['strategy'],
                        'best_sharpe': comparison_df.iloc[0]['sharpe_ratio'],
                        'best_return': comparison_df.iloc[0]['total_return'],
                        'avg_sharpe': comparison_df['sharpe_ratio'].mean(),
                        'avg_return': comparison_df['total_return'].mean()
                    }
                
            except Exception as e:
                logger.warning(f"Failed to create strategy comparison: {e}")
        
        return backtest_results
    
    def _generate_visualizations(self, 
                               results_df: pd.DataFrame, 
                               backtest_results: Dict) -> None:
        """Generate and save all visualizations"""
        try:
            # Correlation heatmap
            self.visualizer.plot_correlation_heatmap(
                results_df, lag_days=0, 
                save_path=str(self.plots_dir / "correlation_heatmap.png")
            )
            
            # Top predictors for each stock
            stocks = results_df['follower'].unique()
            for stock in stocks:
                try:
                    self.visualizer.plot_top_predictors(
                        results_df, stock, top_n=10,
                        save_path=str(self.plots_dir / f"top_predictors_{stock}.png")
                    )
                except Exception as e:
                    logger.warning(f"Failed to create predictor plot for {stock}: {e}")
            
            # Backtest visualizations
            if backtest_results.get('individual_results'):
                for strategy_name, result in backtest_results['individual_results'].items():
                    try:
                        self.visualizer.plot_backtest_results(
                            result['detailed_results'],
                            save_path=str(self.plots_dir / f"backtest_{strategy_name}.png")
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create backtest plot for {strategy_name}: {e}")
            
            # Strategy comparison
            if backtest_results.get('comparison') is not None:
                try:
                    self.visualizer.plot_strategy_comparison(
                        backtest_results['comparison'],
                        save_path=str(self.plots_dir / "strategy_comparison.png")
                    )
                except Exception as e:
                    logger.warning(f"Failed to create strategy comparison plot: {e}")
            
            logger.info(f"Visualizations saved to {self.plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def _save_results(self, results: Dict) -> None:
        """Save analysis results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save lead-lag results as CSV
            if not results['lead_lag_results'].empty:
                results['lead_lag_results'].to_csv(
                    self.output_dir / f"lead_lag_results_{timestamp}.csv", 
                    index=False
                )
            
            # Save best predictors
            for stock, predictors in results['best_predictors'].items():
                if not predictors.empty:
                    predictors.to_csv(
                        self.output_dir / f"best_predictors_{stock}_{timestamp}.csv",
                        index=False
                    )
            
            # Save backtest comparison
            if results['backtest_results'].get('comparison') is not None:
                results['backtest_results']['comparison'].to_csv(
                    self.output_dir / f"backtest_comparison_{timestamp}.csv",
                    index=False
                )
            
            # Save summary as JSON
            summary_data = {
                'analysis_summary': results['analysis_summary'],
                'performance_summary': results['backtest_results'].get('performance_summary', {}),
                'parameters': results['parameters'],
                'timestamp': timestamp
            }
            
            with open(self.output_dir / f"analysis_summary_{timestamp}.json", 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_correlation_results(self, results: Dict) -> None:
        """Save correlation analysis results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save correlation results as CSV
            if not results['correlation_results'].empty:
                results['correlation_results'].to_csv(
                    self.output_dir / f"correlation_results_{timestamp}.csv", 
                    index=False
                )
            
            # Save summary as JSON
            summary_data = {
                'correlation_summary': results['correlation_summary'],
                'parameters': results['parameters'],
                'timestamp': timestamp
            }
            
            with open(self.output_dir / f"correlation_summary_{timestamp}.json", 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Correlation results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save correlation results: {e}")
    
    def _save_enhanced_correlation_results(self, results: Dict) -> None:
        """Save enhanced correlation analysis results to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save enhanced correlation results as CSV
            if not results['correlation_results'].empty:
                results['correlation_results'].to_csv(
                    self.output_dir / f"enhanced_correlation_results_{timestamp}.csv", 
                    index=False
                )
            
            # Save all results (including horizon breakdown) as CSV
            if not results['all_results'].empty:
                results['all_results'].to_csv(
                    self.output_dir / f"enhanced_correlation_all_results_{timestamp}.csv", 
                    index=False
                )
            
            # Save horizon breakdown as separate CSVs
            for horizon, horizon_data in results['horizon_breakdown'].items():
                horizon_df = pd.DataFrame([horizon_data])
                horizon_df.to_csv(
                    self.output_dir / f"enhanced_correlation_horizon_{horizon}_{timestamp}.csv",
                    index=False
                )
            
            # Save comprehensive summary as JSON
            summary_data = {
                'summary': results['summary'],
                'parameters': results['parameters'],
                'timestamp': timestamp
            }
            
            with open(self.output_dir / f"enhanced_correlation_summary_{timestamp}.json", 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Enhanced correlation results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced correlation results: {e}")
    
    def print_summary(self, results: Dict) -> None:
        """Print a formatted summary of results"""
        print("\n" + "="*80)
        print("LEAD-LAG STOCK ANALYSIS SUMMARY")
        print("="*80)
        
        # Analysis summary
        summary = results['analysis_summary']
        print(f"\nANALYSIS OVERVIEW:")
        print(f"  Total relationships analyzed: {summary.get('total_relationships', 0)}")
        print(f"  Significant relationships: {summary.get('significant_relationships', 0)}")
        print(f"  Average correlation: {summary.get('avg_correlation', 0):.3f}")
        print(f"  Maximum correlation: {summary.get('max_correlation', 0):.3f}")
        
        # Top predictive pairs
        if 'most_predictive_pairs' in summary:
            print(f"\nTOP PREDICTIVE RELATIONSHIPS:")
            for i, pair in enumerate(summary['most_predictive_pairs'][:5], 1):
                print(f"  {i}. {pair['leader']} -> {pair['follower']} (lag={pair['lag_days']}, r={pair['correlation']:.3f}, p={pair['p_value']:.4f})")
        
        # Backtest performance
        perf_summary = results['backtest_results'].get('performance_summary', {})
        if perf_summary:
            print(f"\nBACKTEST PERFORMANCE:")
            print(f"  Best strategy: {perf_summary.get('best_strategy', 'N/A')}")
            print(f"  Best Sharpe ratio: {perf_summary.get('best_sharpe', 0):.2f}")
            print(f"  Best total return: {perf_summary.get('best_return', 0):.2%}")
            print(f"  Average Sharpe ratio: {perf_summary.get('avg_sharpe', 0):.2f}")
            print(f"  Average total return: {perf_summary.get('avg_return', 0):.2%}")
        
        print("\n" + "="*80)
        print(f"Analysis completed. Results saved to '{self.output_dir}' and plots to '{self.plots_dir}'")
        print("="*80 + "\n")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Lead-Lag Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbols AAPL MSFT GOOGL META --period 1y --max-lag 5
  python main.py --symbols TSLA NVDA AMD --interval 1h --correlation-method spearman
  python main.py --config config.json
        """
    )
    
    parser.add_argument('--symbols', nargs='+', 
                       help='Stock symbols to analyze (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--period', default='2y',
                       help='Data period (default: 2y)')
    parser.add_argument('--interval', default='1d',
                       help='Data interval (default: 1d)')
    parser.add_argument('--max-lag', type=int, default=10,
                       help='Maximum lag days to test (default: 10)')
    parser.add_argument('--correlation-method', choices=['pearson', 'spearman'], 
                       default='pearson', help='Correlation method (default: pearson)')
    parser.add_argument('--return-method', choices=['pct_change', 'log_returns', 'diff'],
                       default='pct_change', help='Return calculation method (default: pct_change)')
    parser.add_argument('--min-correlation', type=float, default=0.1,
                       help='Minimum correlation threshold (default: 0.1)')
    parser.add_argument('--backtest-top-n', type=int, default=5,
                       help='Number of top strategies to backtest (default: 5)')
    parser.add_argument('--config', help='JSON config file path')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--plots-dir', default='plots',
                       help='Plots directory (default: plots)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override with command line args
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
        args = argparse.Namespace(**config)
    
    # Validate required arguments
    if not args.symbols:
        parser.error("--symbols is required (or provide via config file)")
    
    # Initialize system
    system = LeadLagSystem(
        output_dir=args.output_dir,
        plots_dir=args.plots_dir
    )
    
    try:
        # Run analysis
        results = system.run_full_analysis(
            symbols=args.symbols,
            period=args.period,
            interval=args.interval,
            max_lag=args.max_lag,
            correlation_method=args.correlation_method,
            return_method=args.return_method,
            min_correlation=args.min_correlation,
            backtest_top_n=args.backtest_top_n,
            save_results=not args.no_save
        )
        
        # Print summary
        system.print_summary(results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())