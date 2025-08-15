import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional

# Import our custom modules
from data_fetcher import StockDataFetcher
from lead_lag_analyzer import LeadLagAnalyzer
from backtester import LeadLagBacktester
from visualizer import LeadLagVisualizer
from main import LeadLagSystem

# Configure page
st.set_page_config(
    page_title="Lead-Lag Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging to suppress info messages in Streamlit
logging.getLogger().setLevel(logging.WARNING)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_symbols():
    """Load sample stock symbols"""
    return {
        'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
        'Banking': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD']
    }

@st.cache_resource
def initialize_system():
    """Initialize the lead-lag analysis system"""
    return LeadLagSystem()

def create_correlation_heatmap(results_df, lag_days=0):
    """Create interactive correlation heatmap"""
    lag_data = results_df[results_df['lag_days'] == lag_days]
    
    if lag_data.empty:
        return None
    
    # Create pivot table
    heatmap_data = lag_data.pivot(index='leader', columns='follower', values='correlation')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(heatmap_data.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Lead-Lag Correlation Matrix (Lag: {lag_days} days)',
        xaxis_title='Follower Stock',
        yaxis_title='Leader Stock',
        height=500
    )
    
    return fig

def create_lag_profile_plot(results_df, leader, follower):
    """Create lag correlation profile plot"""
    pair_data = results_df[
        (results_df['leader'] == leader) & 
        (results_df['follower'] == follower)
    ].sort_values('lag_days')
    
    if pair_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Correlation vs Lag', 'Statistical Significance'],
        vertical_spacing=0.1
    )
    
    # Correlation plot
    fig.add_trace(
        go.Scatter(
            x=pair_data['lag_days'],
            y=pair_data['correlation'],
            mode='lines+markers',
            name='Correlation',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=pair_data['lag_days'],
            y=pair_data['ci_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=pair_data['lag_days'],
            y=pair_data['ci_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,100,80,0.2)',
            fill='tonexty',
            name='95% CI',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # P-value plot
    fig.add_trace(
        go.Scatter(
            x=pair_data['lag_days'],
            y=pair_data['p_value'],
            mode='lines+markers',
            name='P-value',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # Significance thresholds
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                 annotation_text="p=0.05", row=2, col=1)
    fig.add_hline(y=0.01, line_dash="dot", line_color="red", 
                 annotation_text="p=0.01", row=2, col=1)
    
    fig.update_layout(
        title=f'Lead-Lag Analysis: {leader} → {follower}',
        height=600
    )
    
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="P-value", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Lag Days", row=2, col=1)
    
    return fig

def create_backtest_plot(backtest_data):
    """Create backtest results plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Portfolio Value', 'Drawdown', 'Returns Distribution', 'Trading Signals'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Drawdown
    portfolio_values = backtest_data['portfolio_value']
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red', width=1)
        ),
        row=1, col=2
    )
    
    # Returns distribution
    returns = backtest_data['returns'].dropna()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Trading signals
    fig.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=2, col=2
    )
    
    # Buy/sell signals
    buy_signals = backtest_data[backtest_data['signal'] == 1]
    sell_signals = backtest_data[backtest_data['signal'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ),
            row=2, col=2
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Backtest Results',
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">📈 Lead-Lag Stock Analysis System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes lead-lag relationships between stocks to identify potential trading opportunities.
    It examines how movements in one stock might predict movements in another stock across different time horizons.
    """)
    
    # Initialize system
    system = initialize_system()
    sample_symbols = load_sample_symbols()
    
    # Sidebar configuration
    st.sidebar.header("📊 Analysis Configuration")
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    
    # Preset groups
    selected_group = st.sidebar.selectbox(
        "Choose a preset group:",
        options=["Custom"] + list(sample_symbols.keys())
    )
    
    if selected_group != "Custom":
        default_symbols = sample_symbols[selected_group]
    else:
        default_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Manual symbol input
    symbols_input = st.sidebar.text_input(
        "Enter stock symbols (comma-separated):",
        value=", ".join(default_symbols)
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    period = st.sidebar.selectbox(
        "Data Period:",
        options=["1y", "2y", "5y", "max"],
        index=1
    )
    
    interval = st.sidebar.selectbox(
        "Data Interval:",
        options=["1d", "1wk", "1mo"],
        index=0
    )
    
    # Analysis parameters
    st.sidebar.subheader("📈 Analysis Parameters")
    max_lag = st.sidebar.slider(
        "Maximum Lag (days):",
        min_value=1,
        max_value=30,
        value=10
    )
    
    min_correlation = st.sidebar.slider(
        "Minimum Correlation Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05
    )
    
    correlation_method = st.sidebar.selectbox(
        "Correlation Method:",
        options=["pearson", "spearman"],
        index=0
    )
    
    return_method = st.sidebar.selectbox(
        "Return Calculation:",
        options=["pct_change", "log_returns"],
        index=0
    )
    
    # Advanced Statistical Methods
    st.sidebar.subheader("🔬 Advanced Statistical Methods")
    
    enable_monte_carlo = st.sidebar.checkbox(
        "Enable Monte Carlo Significance Testing",
        value=True,
        help="Perform Monte Carlo simulations to test statistical significance"
    )
    
    monte_carlo_iterations = st.sidebar.slider(
        "Monte Carlo Iterations:",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        disabled=not enable_monte_carlo
    )
    
    enable_bootstrap = st.sidebar.checkbox(
        "Enable Bootstrap Confidence Intervals",
        value=True,
        help="Calculate bootstrap confidence intervals for correlations"
    )
    
    bootstrap_iterations = st.sidebar.slider(
        "Bootstrap Iterations:",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        disabled=not enable_bootstrap
    )
    
    enable_rolling_analysis = st.sidebar.checkbox(
        "Enable Rolling Window Stability Analysis",
        value=True,
        help="Analyze correlation stability over time using rolling windows"
    )
    
    rolling_window_size = st.sidebar.slider(
        "Rolling Window Size (days):",
        min_value=50,
        max_value=500,
        value=252,
        step=10,
        disabled=not enable_rolling_analysis
    )
    
    apply_multiple_correction = st.sidebar.checkbox(
        "Apply Multiple Comparison Correction",
        value=True,
        help="Apply Benjamini-Hochberg FDR correction for multiple testing"
    )
    
    # Data Transformation Options
    st.sidebar.subheader("🔄 Data Transformation")
    
    enable_detrending = st.sidebar.checkbox(
        "Enable Detrending",
        value=False,
        help="Remove trends from price data before analysis"
    )
    
    detrend_method = st.sidebar.selectbox(
        "Detrending Method:",
        options=["linear", "quadratic", "moving_average"],
        index=0,
        disabled=not enable_detrending
    )
    
    enable_market_adjustment = st.sidebar.checkbox(
        "Remove Market-Wide Effects",
        value=False,
        help="Remove market-wide movements using a benchmark index"
    )
    
    market_index = st.sidebar.selectbox(
        "Market Benchmark:",
        options=["SPY", "QQQ", "IWM", "VTI", "^GSPC"],
        index=0,
        disabled=not enable_market_adjustment
    )
    
    market_adjustment_method = st.sidebar.selectbox(
        "Market Adjustment Method:",
        options=["subtract", "beta_adjust", "residual"],
        index=1,
        disabled=not enable_market_adjustment,
        help="subtract: simple return subtraction, beta_adjust: beta-weighted adjustment, residual: regression residuals"
    )
    
    enable_normalization = st.sidebar.checkbox(
        "Enable Data Normalization",
        value=False,
        help="Normalize data for cross-stock comparisons"
    )
    
    normalization_method = st.sidebar.selectbox(
        "Normalization Method:",
        options=["zscore", "minmax", "robust"],
        index=0,
        disabled=not enable_normalization,
        help="zscore: standard normalization, minmax: 0-1 scaling, robust: median-based scaling"
    )
    
    # Backtest parameters
    st.sidebar.subheader("💰 Backtest Parameters")
    backtest_top_n = st.sidebar.slider("Top Strategies to Backtest:", 1, 10, 5)
    
    # Trading Cost Parameters
    st.sidebar.subheader("💸 Trading Costs")
    
    enable_trading_costs = st.sidebar.checkbox(
        "Enable Realistic Trading Costs",
        value=True,
        help="Include transaction costs, bid-ask spreads, and slippage in backtests"
    )
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%):",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        disabled=not enable_trading_costs,
        help="Commission and fees as percentage of trade value"
    ) / 100  # Convert to decimal
    
    bid_ask_spread = st.sidebar.slider(
        "Bid-Ask Spread (%):",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
        disabled=not enable_trading_costs,
        help="Average bid-ask spread as percentage of price"
    ) / 100  # Convert to decimal
    
    slippage = st.sidebar.slider(
        "Market Impact/Slippage (%):",
        min_value=0.0,
        max_value=0.5,
        value=0.02,
        step=0.01,
        disabled=not enable_trading_costs,
        help="Price impact from market orders as percentage of price"
    ) / 100  # Convert to decimal
    
    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    # Main content area
    if run_analysis:
        if len(symbols) < 2:
            st.error("Please enter at least 2 stock symbols.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Running lead-lag analysis..."):
                # Update progress
                status_text.text("Fetching stock data...")
                progress_bar.progress(20)
                
                # Run analysis with advanced parameters
                analysis_config = {
                    'symbols': symbols,
                    'period': period,
                    'interval': interval,
                    'max_lag': max_lag,
                    'correlation_method': correlation_method,
                    'min_correlation': min_correlation,
                    'backtest_top_n': backtest_top_n,
                    'save_results': False,
                    'return_method': return_method,
                    'enable_monte_carlo': enable_monte_carlo,
                    'monte_carlo_iterations': monte_carlo_iterations,
                    'enable_bootstrap': enable_bootstrap,
                    'bootstrap_iterations': bootstrap_iterations,
                    'enable_rolling_analysis': enable_rolling_analysis,
                    'rolling_window_size': rolling_window_size,
                    'apply_multiple_correction': apply_multiple_correction,
                    'enable_detrending': enable_detrending,
                    'detrend_method': detrend_method,
                    'enable_market_adjustment': enable_market_adjustment,
                    'market_index': market_index,
                    'market_adjustment_method': market_adjustment_method,
                    'enable_normalization': enable_normalization,
                    'normalization_method': normalization_method,
                    # Trading cost parameters
                    'transaction_cost': transaction_cost,
                    'bid_ask_spread': bid_ask_spread,
                    'slippage': slippage,
                    'enable_trading_costs': enable_trading_costs
                }
                
                results = system.run_full_analysis(**analysis_config)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['symbols'] = symbols
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        symbols = st.session_state['symbols']
        
        # Analysis Summary
        st.markdown('<h2 class="sub-header">📋 Analysis Summary</h2>', unsafe_allow_html=True)
        
        summary = results['analysis_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Relationships",
                summary.get('total_relationships', 0)
            )
        
        with col2:
            st.metric(
                "Significant Relationships",
                summary.get('significant_relationships', 0)
            )
        
        with col3:
            st.metric(
                "Average Correlation",
                f"{summary.get('avg_correlation', 0):.3f}"
            )
        
        with col4:
            st.metric(
                "Maximum Correlation",
                f"{summary.get('max_correlation', 0):.3f}"
            )
        
        # Top Predictive Pairs
        if 'most_predictive_pairs' in summary and summary['most_predictive_pairs']:
            st.markdown('<h3 class="sub-header">🏆 Top Predictive Relationships</h3>', unsafe_allow_html=True)
            
            top_pairs_df = pd.DataFrame(summary['most_predictive_pairs'])
            
            # Display columns based on available data
            display_columns = ['leader', 'follower', 'lag_days', 'correlation', 'p_value']
            if 'adjusted_p_value' in top_pairs_df.columns:
                display_columns.append('adjusted_p_value')
            if 'monte_carlo_p_value' in top_pairs_df.columns:
                display_columns.append('monte_carlo_p_value')
            if 'bootstrap_ci_lower' in top_pairs_df.columns and 'bootstrap_ci_upper' in top_pairs_df.columns:
                display_columns.extend(['bootstrap_ci_lower', 'bootstrap_ci_upper'])
            
            st.dataframe(
                top_pairs_df[display_columns],
                use_container_width=True
            )
            
            # Statistical significance indicators
            if 'adjusted_p_value' in top_pairs_df.columns:
                significant_after_correction = (top_pairs_df['adjusted_p_value'] < 0.05).sum()
                st.info(f"📊 {significant_after_correction} relationships remain significant after multiple comparison correction")
            
            if 'monte_carlo_p_value' in top_pairs_df.columns:
                monte_carlo_significant = (top_pairs_df['monte_carlo_p_value'] < 0.05).sum()
                st.info(f"🎲 {monte_carlo_significant} relationships confirmed significant by Monte Carlo testing")
        
        # Interactive Visualizations
        st.markdown('<h2 class="sub-header">📊 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        results_df = results['lead_lag_results']
        
        if not results_df.empty:
            # Correlation Heatmap
            st.markdown('<h3 class="sub-header">Correlation Heatmap</h3>', unsafe_allow_html=True)
            
            lag_options = sorted(results_df['lag_days'].unique())
            selected_lag = st.selectbox("Select lag days for heatmap:", lag_options)
            
            heatmap_fig = create_correlation_heatmap(results_df, selected_lag)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Lag Profile Analysis
            st.markdown('<h3 class="sub-header">Lag Profile Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                leader_options = sorted(results_df['leader'].unique())
                selected_leader = st.selectbox("Select leader stock:", leader_options)
            
            with col2:
                follower_options = sorted(results_df[results_df['leader'] == selected_leader]['follower'].unique())
                if follower_options:
                    selected_follower = st.selectbox("Select follower stock:", follower_options)
                    
                    lag_profile_fig = create_lag_profile_plot(results_df, selected_leader, selected_follower)
                    if lag_profile_fig:
                        st.plotly_chart(lag_profile_fig, use_container_width=True)
        
        # Best Predictors for Each Stock
        st.markdown('<h2 class="sub-header">🎯 Best Predictors by Stock</h2>', unsafe_allow_html=True)
        
        best_predictors = results['best_predictors']
        
        if best_predictors:
            selected_stock = st.selectbox(
                "Select stock to view its best predictors:",
                list(best_predictors.keys())
            )
            
            if selected_stock in best_predictors:
                predictors_df = best_predictors[selected_stock]
                st.dataframe(
                    predictors_df[['leader', 'lag_days', 'correlation', 'p_value', 'strength']],
                    use_container_width=True
                )
        
        # Backtest Results
        backtest_results = results['backtest_results']
        
        if backtest_results.get('individual_results'):
            st.markdown('<h2 class="sub-header">💰 Backtest Results</h2>', unsafe_allow_html=True)
            
            # Performance summary
            perf_summary = backtest_results.get('performance_summary', {})
            if perf_summary:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Best Strategy",
                        perf_summary.get('best_strategy', 'N/A')[:20] + "..."
                    )
                
                with col2:
                    st.metric(
                        "Best Sharpe Ratio",
                        f"{perf_summary.get('best_sharpe', 0):.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Best Total Return",
                        f"{perf_summary.get('best_return', 0):.2%}"
                    )
                
                with col4:
                    st.metric(
                        "Average Sharpe",
                        f"{perf_summary.get('avg_sharpe', 0):.2f}"
                    )
            
            # Strategy comparison table
            if backtest_results.get('comparison') is not None:
                st.markdown('<h3 class="sub-header">Strategy Comparison</h3>', unsafe_allow_html=True)
                comparison_df = backtest_results['comparison']
                st.dataframe(
                    comparison_df[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']],
                    use_container_width=True
                )
            
            # Individual backtest plots
            st.markdown('<h3 class="sub-header">Individual Strategy Results</h3>', unsafe_allow_html=True)
            
            strategy_names = list(backtest_results['individual_results'].keys())
            selected_strategy = st.selectbox("Select strategy to view detailed results:", strategy_names)
            
            if selected_strategy:
                strategy_data = backtest_results['individual_results'][selected_strategy]
                detailed_results = strategy_data['detailed_results']
                performance = strategy_data['performance']
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{performance.total_return:.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{performance.sharpe_ratio:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{performance.max_drawdown:.2%}")
                
                with col4:
                    st.metric("Win Rate", f"{performance.win_rate:.2%}")
                
                # Backtest plot
                backtest_fig = create_backtest_plot(detailed_results)
                st.plotly_chart(backtest_fig, use_container_width=True)
        
        # Advanced Analysis Results
        if 'rolling_stability' in results:
            st.markdown('<h2 class="sub-header">📈 Rolling Window Stability Analysis</h2>', unsafe_allow_html=True)
            
            stability_results = results['rolling_stability']
            if stability_results:
                # Create stability plot for top relationship
                top_pair = list(stability_results.keys())[0]
                stability_data = stability_results[top_pair]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stability_data.index,
                    y=stability_data['rolling_correlation'],
                    mode='lines',
                    name=f'Rolling Correlation: {top_pair}',
                    line=dict(width=2)
                ))
                
                # Add confidence bands if available
                if 'rolling_ci_upper' in stability_data.columns:
                    fig.add_trace(go.Scatter(
                        x=stability_data.index,
                        y=stability_data['rolling_ci_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=stability_data.index,
                        y=stability_data['rolling_ci_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.2)',
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                
                fig.update_layout(
                    title=f'Correlation Stability Over Time: {top_pair}',
                    xaxis_title='Date',
                    yaxis_title='Rolling Correlation',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stability metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation Volatility", f"{stability_data['rolling_correlation'].std():.3f}")
                with col2:
                    st.metric("Mean Correlation", f"{stability_data['rolling_correlation'].mean():.3f}")
                with col3:
                    stability_ratio = stability_data['rolling_correlation'].std() / abs(stability_data['rolling_correlation'].mean())
                    st.metric("Stability Ratio", f"{stability_ratio:.3f}")
        
        # Statistical Method Comparison
        if 'method_comparison' in results:
            st.markdown('<h2 class="sub-header">🔬 Statistical Method Comparison</h2>', unsafe_allow_html=True)
            
            comparison_data = results['method_comparison']
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Method agreement analysis
            if len(comparison_df) > 1:
                agreement_score = (comparison_df['significant'].sum() / len(comparison_df)) * 100
                st.info(f"📊 Method Agreement: {agreement_score:.1f}% of methods agree on significance")
        
        # Raw Data Tables
        with st.expander("📋 View Raw Data"):
            st.markdown('<h3 class="sub-header">Lead-Lag Analysis Results</h3>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"lead_lag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Additional data downloads
            if 'rolling_stability' in results and results['rolling_stability']:
                st.markdown('<h3 class="sub-header">Rolling Stability Data</h3>', unsafe_allow_html=True)
                
                # Combine all stability data
                all_stability_data = []
                for pair, data in results['rolling_stability'].items():
                    data_copy = data.copy()
                    data_copy['pair'] = pair
                    all_stability_data.append(data_copy)
                
                if all_stability_data:
                    combined_stability = pd.concat(all_stability_data, ignore_index=True)
                    st.dataframe(combined_stability, use_container_width=True)
                    
                    stability_csv = combined_stability.to_csv(index=False)
                    st.download_button(
                        label="Download Stability Data as CSV",
                        data=stability_csv,
                        file_name=f"stability_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Lead-Lag Stock Analysis System** - Identify predictive relationships between stocks across different time horizons.
    
    💡 **Tips:**
    - Use at least 2 years of data for reliable results
    - Higher correlation thresholds reduce noise but may miss weak signals
    - Monte Carlo testing provides more robust significance testing
    - Bootstrap confidence intervals help assess estimation uncertainty
    - Rolling window analysis reveals relationship stability over time
    - Multiple comparison correction reduces false discovery rate
    - Market adjustment helps isolate stock-specific relationships
    - Consider transaction costs and market impact in real trading
    - Past performance doesn't guarantee future results
    
    🔬 **Advanced Features:**
    - **Monte Carlo Testing**: Simulates random data to test if correlations are statistically significant
    - **Bootstrap Confidence Intervals**: Provides uncertainty estimates for correlation coefficients
    - **Rolling Window Analysis**: Tracks how relationships change over time
    - **Multiple Comparison Correction**: Adjusts p-values when testing many relationships simultaneously
    - **Market Adjustment**: Removes market-wide effects to focus on stock-specific relationships
    - **Data Normalization**: Standardizes data for better cross-stock comparisons
    """)

if __name__ == "__main__":
    main()