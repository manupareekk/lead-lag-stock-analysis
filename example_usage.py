#!/usr/bin/env python3
"""
Example Usage of Lead-Lag Stock Analysis System

This script demonstrates various ways to use the lead-lag analysis system
with practical examples and different use cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import our modules
from main import LeadLagSystem
from data_fetcher import StockDataFetcher
from lead_lag_analyzer import LeadLagAnalyzer
from backtester import LeadLagBacktester
from visualizer import LeadLagVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_analysis():
    """Example 1: Basic lead-lag analysis between tech stocks"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Tech Stock Analysis")
    print("="*60)
    
    # Initialize system
    system = LeadLagSystem()
    
    # Define tech stocks to analyze
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META']
    
    print(f"Analyzing relationships between: {', '.join(tech_stocks)}")
    
    try:
        # Run analysis
        results = system.run_full_analysis(
            symbols=tech_stocks,
            period='1y',  # 1 year of data
            max_lag=5,    # Test up to 5 days lag
            min_correlation=0.2,  # Only show correlations > 0.2
            backtest_top_n=3,     # Backtest top 3 strategies
            save_results=True
        )
        
        # Print summary
        summary = results['analysis_summary']
        print(f"\nFound {summary['significant_relationships']} significant relationships")
        print(f"Average correlation: {summary['avg_correlation']:.3f}")
        print(f"Maximum correlation: {summary['max_correlation']:.3f}")
        
        # Show top relationships
        if 'most_predictive_pairs' in summary:
            print("\nTop 3 Predictive Relationships:")
            for i, pair in enumerate(summary['most_predictive_pairs'][:3], 1):
                print(f"  {i}. {pair['leader']} → {pair['follower']} ")
                print(f"     Lag: {pair['lag_days']} days, Correlation: {pair['correlation']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

def example_2_sector_comparison():
    """Example 2: Compare lead-lag relationships across different sectors"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Sector Comparison")
    print("="*60)
    
    # Define sectors
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Banking': ['JPM', 'BAC', 'WFC'],
        'Energy': ['XOM', 'CVX', 'COP']
    }
    
    system = LeadLagSystem()
    sector_results = {}
    
    for sector_name, symbols in sectors.items():
        print(f"\nAnalyzing {sector_name} sector: {', '.join(symbols)}")
        
        try:
            results = system.run_full_analysis(
                symbols=symbols,
                period='2y',
                max_lag=7,
                min_correlation=0.15,
                backtest_top_n=2,
                save_results=False  # Don't save individual results
            )
            
            sector_results[sector_name] = results
            
            # Print sector summary
            summary = results['analysis_summary']
            print(f"  Significant relationships: {summary['significant_relationships']}")
            print(f"  Average correlation: {summary['avg_correlation']:.3f}")
            
        except Exception as e:
            print(f"  Failed to analyze {sector_name}: {e}")
    
    # Compare sectors
    print("\n" + "-"*40)
    print("SECTOR COMPARISON SUMMARY")
    print("-"*40)
    
    for sector, results in sector_results.items():
        summary = results['analysis_summary']
        backtest = results['backtest_results'].get('performance_summary', {})
        
        print(f"\n{sector}:")
        print(f"  Relationships: {summary['significant_relationships']}")
        print(f"  Avg Correlation: {summary['avg_correlation']:.3f}")
        if backtest:
            print(f"  Best Sharpe: {backtest.get('best_sharpe', 0):.2f}")
            print(f"  Best Return: {backtest.get('best_return', 0):.2%}")
    
    return sector_results

def example_3_custom_analysis():
    """Example 3: Custom analysis with specific parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Analysis - High Frequency")
    print("="*60)
    
    # Initialize components separately for custom analysis
    fetcher = StockDataFetcher()
    analyzer = LeadLagAnalyzer(min_periods=20)
    backtester = LeadLagBacktester(initial_capital=50000, transaction_cost=0.002)
    
    # Focus on two specific stocks with hourly data
    symbols = ['AAPL', 'MSFT']
    
    print(f"Custom analysis: {symbols[0]} vs {symbols[1]} with hourly data")
    
    try:
        # Get hourly data for last month
        price_data = fetcher.get_price_data(
            symbols, 
            period='1mo', 
            interval='1h'
        )
        
        print(f"Retrieved {len(price_data)} hourly observations")
        
        # Analyze with custom parameters
        results_df = analyzer.analyze_multiple_pairs(
            price_data,
            max_lag=24,  # 24 hours
            method='spearman',  # Use Spearman correlation
            return_method='log_returns',
            min_correlation=0.05
        )
        
        if not results_df.empty:
            print(f"\nFound {len(results_df)} relationships")
            
            # Show best relationship
            best = results_df.iloc[0]
            print(f"\nBest relationship:")
            print(f"  {best['leader']} → {best['follower']}")
            print(f"  Lag: {best['lag_days']} hours")
            print(f"  Correlation: {best['correlation']:.3f}")
            print(f"  P-value: {best['p_value']:.4f}")
            
            # Custom backtest
            if best['lag_days'] > 0:
                performance, detailed = backtester.backtest_strategy(
                    leader_data=price_data[best['leader']],
                    follower_data=price_data[best['follower']],
                    lag_days=int(best['lag_days']),
                    threshold=0.01,  # 1% threshold for hourly data
                    max_position_size=0.2
                )
                
                print(f"\nBacktest Results:")
                print(f"  Total Return: {performance.total_return:.2%}")
                print(f"  Sharpe Ratio: {performance.sharpe_ratio:.2f}")
                print(f"  Max Drawdown: {performance.max_drawdown:.2%}")
                print(f"  Total Trades: {performance.total_trades}")
        
        else:
            print("No significant relationships found")
            
    except Exception as e:
        print(f"Custom analysis failed: {e}")

def example_4_specific_hypothesis():
    """Example 4: Test a specific hypothesis"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Hypothesis Testing")
    print("="*60)
    
    # Hypothesis: "Tesla movements predict other EV/tech stocks"
    leader = 'TSLA'
    followers = ['NVDA', 'AMD', 'AAPL']  # Tech stocks that might follow Tesla
    
    print(f"Hypothesis: {leader} movements predict {', '.join(followers)}")
    
    fetcher = StockDataFetcher()
    analyzer = LeadLagAnalyzer()
    
    try:
        # Get data
        all_symbols = [leader] + followers
        price_data = fetcher.get_price_data(all_symbols, period='1y')
        
        print(f"Testing {leader} as leader against {len(followers)} followers")
        
        # Test each follower
        hypothesis_results = []
        
        for follower in followers:
            print(f"\nTesting {leader} → {follower}:")
            
            # Analyze this specific pair
            pair_results = analyzer.analyze_lead_lag_pair(
                price_data[leader],
                price_data[follower],
                leader,
                follower,
                max_lag=10
            )
            
            # Find best lag
            if pair_results:
                best_result = max(pair_results, key=lambda x: abs(x.correlation))
                
                print(f"  Best lag: {best_result.lag_days} days")
                print(f"  Correlation: {best_result.correlation:.3f}")
                print(f"  P-value: {best_result.p_value:.4f}")
                print(f"  Significant: {'Yes' if best_result.is_significant else 'No'}")
                print(f"  Strength: {best_result.strength_category}")
                
                hypothesis_results.append({
                    'follower': follower,
                    'best_lag': best_result.lag_days,
                    'correlation': best_result.correlation,
                    'p_value': best_result.p_value,
                    'significant': best_result.is_significant
                })
        
        # Summary of hypothesis test
        print(f"\n" + "-"*40)
        print("HYPOTHESIS TEST SUMMARY")
        print("-"*40)
        
        significant_count = sum(1 for r in hypothesis_results if r['significant'])
        print(f"Significant relationships: {significant_count}/{len(followers)}")
        
        if significant_count > 0:
            print("\nSupported relationships:")
            for result in hypothesis_results:
                if result['significant']:
                    print(f"  {leader} → {result['follower']}: "
                          f"r={result['correlation']:.3f}, lag={result['best_lag']} days")
            
            print(f"\n✅ Hypothesis partially supported ({significant_count}/{len(followers)} relationships)")
        else:
            print("\n❌ Hypothesis not supported (no significant relationships found)")
            
    except Exception as e:
        print(f"Hypothesis test failed: {e}")

def example_5_portfolio_strategy():
    """Example 5: Build a portfolio strategy based on lead-lag relationships"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Portfolio Strategy Development")
    print("="*60)
    
    # Portfolio of diverse stocks
    portfolio_stocks = ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT']  # ETFs for diversification
    
    print(f"Developing portfolio strategy with: {', '.join(portfolio_stocks)}")
    
    system = LeadLagSystem()
    
    try:
        # Run comprehensive analysis
        results = system.run_full_analysis(
            symbols=portfolio_stocks,
            period='3y',  # Longer period for stability
            max_lag=15,   # Test longer lags
            min_correlation=0.1,
            backtest_top_n=8,
            save_results=True
        )
        
        # Analyze results for portfolio construction
        lead_lag_df = results['lead_lag_results']
        backtest_results = results['backtest_results']
        
        print(f"\nPortfolio Analysis Results:")
        print(f"Total relationships analyzed: {len(lead_lag_df)}")
        
        # Find the most reliable predictors
        reliable_predictors = lead_lag_df[
            (lead_lag_df['is_significant']) & 
            (lead_lag_df['abs_correlation'] > 0.3) &
            (lead_lag_df['p_value'] < 0.01)  # Very significant
        ].sort_values('abs_correlation', ascending=False)
        
        print(f"Highly reliable relationships: {len(reliable_predictors)}")
        
        if not reliable_predictors.empty:
            print("\nTop 5 Most Reliable Predictors:")
            for i, (_, row) in enumerate(reliable_predictors.head().iterrows(), 1):
                print(f"  {i}. {row['leader']} → {row['follower']} "
                      f"(lag={row['lag_days']}, r={row['correlation']:.3f})")
        
        # Strategy performance summary
        if backtest_results.get('comparison') is not None:
            comparison = backtest_results['comparison']
            
            print(f"\nStrategy Performance Summary:")
            print(f"Best strategy: {comparison.iloc[0]['strategy']}")
            print(f"Best Sharpe ratio: {comparison.iloc[0]['sharpe_ratio']:.2f}")
            print(f"Average return: {comparison['total_return'].mean():.2%}")
            
            # Risk analysis
            avg_drawdown = comparison['max_drawdown'].mean()
            print(f"Average max drawdown: {avg_drawdown:.2%}")
            
            if avg_drawdown < -0.15:  # More than 15% drawdown
                print("⚠️  High risk strategies - consider position sizing")
            else:
                print("✅ Reasonable risk levels")
        
        print(f"\n📊 Detailed results saved to output/ directory")
        print(f"📈 Visualizations saved to plots/ directory")
        
        return results
        
    except Exception as e:
        print(f"Portfolio strategy development failed: {e}")
        return None

def main():
    """Run all examples"""
    print("🚀 Lead-Lag Stock Analysis System - Example Usage")
    print("=" * 80)
    
    examples = [
        ("Basic Tech Stock Analysis", example_1_basic_analysis),
        ("Multi-Sector Comparison", example_2_sector_comparison),
        ("Custom High-Frequency Analysis", example_3_custom_analysis),
        ("Hypothesis Testing", example_4_specific_hypothesis),
        ("Portfolio Strategy Development", example_5_portfolio_strategy)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} Running: {name} {'='*20}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Final summary
    print("\n" + "="*80)
    print("EXAMPLE EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(examples)
    
    print(f"Successfully completed: {successful}/{total} examples")
    
    if successful > 0:
        print("\n✅ Examples completed! Check the output/ and plots/ directories for results.")
        print("\n💡 Next steps:")
        print("   1. Review the generated plots and analysis results")
        print("   2. Try the Streamlit web interface: streamlit run streamlit_app.py")
        print("   3. Experiment with your own stock symbols and parameters")
        print("   4. Consider the statistical significance and practical implications")
    else:
        print("\n❌ No examples completed successfully. Check your internet connection and try again.")
    
    print("\n📚 For more information, see README.md")
    print("⚠️  Remember: This is for educational purposes only, not financial advice!")

if __name__ == "__main__":
    main()