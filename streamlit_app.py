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
from performance_monitor import performance_monitor

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
    """Load sample stock symbols with comprehensive coverage from available cache data"""
    return {
        'Technology': [
            # Mega-cap tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'TSLA', 'NVDA',
            # Software & Cloud
            'CRM', 'ORCL', 'ADBE', 'NOW', 'INTU', 'WDAY', 'SNOW', 'PLTR', 'DDOG', 'ZM',
            'OKTA', 'CRWD', 'NET', 'TEAM', 'VEEV', 'ZS', 'PANW',
            # Semiconductors
            'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL', 'LRCX', 'KLAC', 'AMAT',
            'MU', 'NXPI', 'MCHP', 'ON', 'SWKS', 'QRVO', 'MPWR', 'ENPH', 'SEDG',
            # Hardware & Electronics
            'HPQ', 'DELL', 'WDC', 'STX', 'NTAP', 'PSTG', 'PURE'
        ],
        'Healthcare': [
            # Pharmaceuticals
            'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB', 'VRTX',
            'REGN', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'TAK', 'NVO', 'SNY',
            # Medical Devices & Healthcare Services
            'UNH', 'CVS', 'CI', 'HUM', 'CNC', 'MOH',
            'MDT', 'ABT', 'TMO', 'DHR', 'SYK', 'BSX', 'EW', 'ISRG', 'DXCM',
            'HOLX', 'BDX', 'BAX', 'ZBH', 'ALGN', 'IDXX', 'IQV', 'MTD', 'A', 'WST',
            # Biotech
            'BMRN', 'TECH', 'INCY', 'EXAS', 'IONS', 'ALNY', 'RARE'
        ],
        'Financial': [
            # Major Banks
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
            'BK', 'STT', 'NTRS', 'RF', 'CFG', 'KEY', 'FITB', 'HBAN', 'ZION', 'CMA',
            # Insurance
            'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'PGR', 'CB', 'AXP',
            'MMC', 'AON', 'WTW', 'BRO', 'AJG', 'HIG', 'LNC', 'UNM', 'RGA', 'AFG',
            # Asset Management & REITs
            'BLK', 'SCHW', 'TROW', 'BEN', 'IVZ', 'AMG',
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'VTR'
        ],
        'Energy': [
            # Oil & Gas
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'OXY', 'FANG',
            'MPC', 'VLO', 'PSX', 'HES', 'DVN', 'APA', 'OVV', 'CTRA', 'EQT',
            'AR', 'SM', 'RRC', 'CNX', 'MTDR',
            # Renewable Energy
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ED', 'ES',
            'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'CSIQ', 'JKS', 'DQ', 'SOL'
        ],
        'Consumer Discretionary': [
            # Retail
            'AMZN', 'WMT', 'HD', 'LOW', 'TGT', 'COST', 'TJX', 'SBUX', 'MCD', 'NKE',
            'LULU', 'ROST', 'DG', 'DLTR', 'BBY', 'ANF', 'AEO', 'URBN',
            # Automotive
            'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'GOEV',
            # Entertainment & Media
            'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'CHTR', 'FOXA', 'PARA', 'WBD',
            'SPOT', 'ROKU', 'FUBO', 'SIRI', 'LBRDK', 'LBRDA', 'NWSA'
        ],
        'Consumer Staples': [
            # Food & Beverages
            'WMT', 'COST', 'CVS',
            # Note: Many consumer staples stocks not available in cache
        ],
        'Industrials': [
            # Aerospace & Defense
            'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'LDOS', 'HII', 'KTOS',
            # Manufacturing
            'GE', 'HON', 'MMM', 'CAT', 'DE', 'EMR', 'ITW', 'PH', 'ROK', 'DOV',
            'ETN', 'JCI', 'CMI', 'FTV', 'AME', 'ROP', 'IEX', 'XYL', 'FLS', 'PNR',
            # Transportation
            'UPS', 'FDX', 'DAL', 'UAL', 'AAL', 'LUV', 'JBLU',
            'UNP', 'CSX', 'NSC', 'CP', 'CNI', 'CHRW', 'EXPD', 'XPO', 'ODFL'
        ],
        'Materials': [
            # Note: Limited materials stocks in cache
        ],
        'Utilities': [
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ED', 'ES'
        ],
        'Real Estate': [
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'VTR'
        ],
        'All Available Stocks': [
            # All stocks from cache directory
            'AAL', 'AAPL', 'ABBV', 'ABT', 'ADBE', 'ADI', 'AEO', 'AEP', 'AFG', 'AFL',
            'AIG', 'AJG', 'ALGN', 'ALK', 'ALL', 'ALNY', 'AMAT', 'AMD', 'AME', 'AMGN',
            'AMG', 'AMT', 'AMZN', 'ANF', 'AON', 'APA', 'AR', 'AVB', 'AVGO', 'AXP',
            'A', 'BAC', 'BAX', 'BA', 'BBY', 'BDX', 'BEN', 'BIIB', 'BKR', 'BK',
            'BLK', 'BMRN', 'BMY', 'BNTX', 'BRO', 'BSX', 'CAT', 'CB', 'CCI', 'CFG',
            'CHRW', 'CHTR', 'CI', 'CMA', 'CMCSA', 'CMI', 'CNC', 'CNI', 'CNX', 'COF',
            'COP', 'COST', 'CP', 'CRM', 'CRWD', 'CSIQ', 'CSX', 'CTRA', 'CVS', 'CVX',
            'C', 'DAL', 'DDOG', 'DELL', 'DE', 'DG', 'DHR', 'DIS', 'DLTR', 'DOV',
            'DQ', 'DUK', 'DVN', 'DXCM', 'D', 'ED', 'EMR', 'ENPH', 'EOG', 'EQIX',
            'EQR', 'EQT', 'ES', 'ETN', 'EW', 'EXAS', 'EXC', 'EXPD', 'FANG', 'FDX',
            'FITB', 'FLS', 'FOXA', 'FSLR', 'FTV', 'FUBO', 'F', 'GD', 'GE', 'GILD',
            'GM', 'GOEV', 'GOOGL', 'GOOG', 'GS', 'HAL', 'HBAN', 'HD', 'HES', 'HIG',
            'HII', 'HOLX', 'HON', 'HPQ', 'HUM', 'IDXX', 'IEX', 'ILMN', 'INCY', 'INTC',
            'INTU', 'IONS', 'IQV', 'ISRG', 'ITW', 'IVZ', 'JBLU', 'JCI', 'JKS', 'JNJ',
            'JPM', 'KEY', 'KLAC', 'KTOS', 'LBRDA', 'LBRDK', 'LCID', 'LDOS', 'LHX', 'LI',
            'LLY', 'LMT', 'LNC', 'LOW', 'LRCX', 'LULU', 'LUV', 'MCD', 'MCHP', 'MDT',
            'META', 'MET', 'MMC', 'MMM', 'MOH', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRVL',
            'MSFT', 'MS', 'MTDR', 'MTD', 'MU', 'NEE', 'NET', 'NFLX', 'NIO', 'NKE',
            'NOC', 'NOW', 'NSC', 'NTAP', 'NTRS', 'NVDA', 'NVO', 'NWSA', 'NXPI', 'ODFL',
            'OKTA', 'ON', 'ORCL', 'OVV', 'OXY', 'O', 'PANW', 'PARA', 'PFE', 'PGR',
            'PH', 'PLD', 'PLTR', 'PNC', 'PNR', 'PRU', 'PSTG', 'PSX', 'PURE', 'QCOM',
            'QRVO', 'RARE', 'REGN', 'RF', 'RGA', 'RIVN', 'ROKU', 'ROK', 'ROP', 'ROST',
            'RRC', 'RTX', 'RUN', 'SBUX', 'SCHW', 'SEDG', 'SIRI', 'SLB', 'SM', 'SNOW',
            'SNY', 'SOL', 'SO', 'SPG', 'SPOT', 'SPWR', 'SPY', 'SRE', 'STT', 'STX',
            'SWKS', 'SYK', 'TAK', 'TDG', 'TEAM', 'TECH', 'TFC', 'TGT', 'TJX', 'TMO',
            'TROW', 'TRV', 'TSLA', 'TXN', 'T', 'UAL', 'UNH', 'UNM', 'UNP', 'UPS',
            'URBN', 'USB', 'VEEV', 'VLO', 'VRTX', 'VTR', 'VZ', 'WBD', 'WDAY', 'WDC',
            'WELL', 'WFC', 'WMT', 'WST', 'WTW', 'XEL', 'XOM', 'XPEV', 'XPO', 'XYL',
            'ZBH', 'ZION', 'ZM', 'ZS', 'ZTS'
        ]
    }

@st.cache_resource
def initialize_system():
    """Initialize the lead-lag analysis system"""
    return LeadLagSystem()

def get_time_unit_label(interval):
    """Get appropriate time unit label based on interval"""
    if interval == "1m":
        return "minutes"
    elif interval in ["5m", "15m", "30m"]:
        return "minutes"
    elif interval == "1h":
        return "hours"
    else:
        return "days"

def create_correlation_heatmap(results_df, lag_periods=0, interval='1d'):
    """Create interactive correlation heatmap"""
    lag_data = results_df[results_df['lag_days'] == lag_periods]
    
    if lag_data.empty:
        return None
    
    # Get appropriate unit label
    unit_label = get_time_unit_label(interval)
    
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
        title=f'Lead-Lag Correlation Matrix (Lag: {lag_periods} {unit_label})',
        xaxis_title='Follower Stock',
        yaxis_title='Leader Stock',
        height=500
    )
    
    return fig

def create_lag_profile_plot(results_df, leader, follower, interval='1d'):
    """Create lag correlation profile plot"""
    pair_data = results_df[
        (results_df['leader'] == leader) & 
        (results_df['follower'] == follower)
    ].sort_values('lag_days')
    
    if pair_data.empty:
        return None
    
    # Get appropriate unit label (capitalize first letter for display)
    unit_label = get_time_unit_label(interval).capitalize()
    
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
    fig.update_xaxes(title_text=f"Lag {unit_label}", row=2, col=1)
    
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
    
    # Analysis type selection with visual indicators
    st.sidebar.subheader("🎯 Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Choose analysis type:",
        options=["Lead-Lag Analysis", "Correlation Discovery", "Enhanced Correlation Discovery"],
        index=0,
        help="Lead-Lag Analysis: Find predictive relationships between stocks. Correlation Discovery: Find most correlated stocks in the universe. Enhanced Correlation Discovery: Advanced analysis across multiple time horizons and lag periods."
    )
    
    # Show analysis type description with emoji indicators
    analysis_descriptions = {
        "Lead-Lag Analysis": "🔍 **Lead-Lag Analysis** - Identifies predictive relationships between stock pairs",
        "Correlation Discovery": "🌐 **Correlation Discovery** - Finds highly correlated stocks across the market",
        "Enhanced Correlation Discovery": "🚀 **Enhanced Correlation Discovery** - Advanced multi-horizon correlation analysis"
    }
    
    st.sidebar.markdown(analysis_descriptions[analysis_type])
    st.sidebar.markdown("---")
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    period = st.sidebar.selectbox(
        "Data Period:",
        options=["1w", "1mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,  # Default to 1y
        help="Time period for historical data analysis"
    )
    
    interval = st.sidebar.selectbox(
        "Data Interval:",
        options=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
        index=5,  # Default to 1d
        help="Granularity of price data (high-frequency intervals limited to shorter periods)"
    )
    
    # Show warning for high-frequency data with long periods
    if interval in ["1m", "5m", "15m", "30m", "1h"] and period in ["2y", "5y", "max"]:
        st.sidebar.warning(
            "⚠️ High-frequency data (1m-1h) is only available for shorter periods. "
            "Yahoo Finance limits: 1m data ≤ 7 days, sub-hourly data ≤ 60 days, 1h data ≤ 730 days."
        )
    
    # Context-aware analysis parameters based on selected analysis type
    if analysis_type == "Lead-Lag Analysis":
        st.sidebar.subheader("🔍 Lead-Lag Parameters")
        
        # Adjust lag parameters based on interval
        if interval == "1m":
            lag_unit = "minutes"
            max_lag_default = 30  # 30 minutes
            max_lag_max = 120     # 2 hours in minutes
            lag_help = "Maximum lag in minutes for 1-minute data"
        elif interval in ["5m", "15m", "30m"]:
            lag_unit = "minutes"
            max_lag_default = 120  # 2 hours in minutes
            max_lag_max = 720      # 12 hours in minutes
            lag_help = f"Maximum lag in minutes for {interval} data"
        elif interval == "1h":
            lag_unit = "hours"
            max_lag_default = 24  # 1 day in hours
            max_lag_max = 168     # 1 week in hours
            lag_help = "Maximum lag in hours for 1-hour data"
        else:
            lag_unit = "days"
            max_lag_default = 10
            max_lag_max = 30
            lag_help = "Maximum lag in days for daily+ data"
        
        max_lag = st.sidebar.slider(
            f"Maximum Lag ({lag_unit}):",
            min_value=1,
            max_value=max_lag_max,
            value=max_lag_default,
            help=lag_help
        )
        
        min_correlation = st.sidebar.slider(
            "Minimum Correlation Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Only show lead-lag relationships above this correlation threshold"
        )
        
    elif analysis_type in ["Correlation Discovery", "Enhanced Correlation Discovery"]:
        st.sidebar.subheader("🌐 Correlation Parameters")
        
        min_correlation = st.sidebar.slider(
            "Minimum Correlation Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Only show correlations above this threshold"
        )
        
        # For Enhanced Correlation Discovery, we don't need max_lag as it uses predefined lag periods
        if analysis_type == "Enhanced Correlation Discovery":
            st.sidebar.info("💡 Lag periods are configured in the Enhanced Parameters section below")
        else:
            # Simple max lag for basic correlation discovery
            max_lag = st.sidebar.slider(
                "Maximum Lag (days):",
                min_value=1,
                max_value=30,
                value=5,
                help="Maximum lag in days for correlation discovery"
            )
    
    # Common parameters for all analysis types
    st.sidebar.subheader("📊 Common Parameters")
    
    correlation_method = st.sidebar.selectbox(
        "Correlation Method:",
        options=["pearson", "spearman"],
        index=0,
        help="Pearson: linear relationships, Spearman: monotonic relationships"
    )
    
    return_method = st.sidebar.selectbox(
        "Return Calculation:",
        options=["pct_change", "log_returns"],
        index=0,
        help="Method for calculating stock returns"
    )
    
    # Advanced Statistical Methods - context-aware
    if analysis_type == "Lead-Lag Analysis":
        st.sidebar.subheader("🔬 Lead-Lag Statistical Methods")
        
        enable_monte_carlo = st.sidebar.checkbox(
            "Enable Monte Carlo Significance Testing",
            value=True,
            help="Test if lead-lag relationships are statistically significant"
        )
        
        monte_carlo_iterations = st.sidebar.slider(
            "Monte Carlo Iterations:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            disabled=not enable_monte_carlo,
            help="Number of random permutations for significance testing"
        )
        
        enable_bootstrap = st.sidebar.checkbox(
            "Enable Bootstrap Confidence Intervals",
            value=True,
            help="Calculate confidence intervals for lead-lag correlations"
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
            help="Analyze lead-lag relationship stability over time"
        )
        
    elif analysis_type in ["Correlation Discovery", "Enhanced Correlation Discovery"]:
        st.sidebar.subheader("🔬 Correlation Statistical Methods")
        
        enable_monte_carlo = st.sidebar.checkbox(
            "Enable Monte Carlo Significance Testing",
            value=True,
            help="Test if correlations are statistically significant"
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
            help="Calculate confidence intervals for correlations"
        )
        
        bootstrap_iterations = st.sidebar.slider(
            "Bootstrap Iterations:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            disabled=not enable_bootstrap
        )
        
        # Rolling analysis is less relevant for correlation discovery
        enable_rolling_analysis = st.sidebar.checkbox(
            "Enable Rolling Window Analysis",
            value=False,
            help="Analyze correlation stability over time (optional for discovery)"
        )
    
    # Adjust rolling window parameters based on interval
    if interval == "1m":
        window_unit = "minutes"
        window_default = 120   # 2 hours in minutes
        window_min = 30        # 30 minutes
        window_max = 480       # 8 hours
        window_step = 30
        window_help = "Rolling window size in minutes for stability analysis"
    elif interval in ["5m", "15m", "30m"]:
        window_unit = "minutes"
        window_default = 240   # 4 hours in minutes
        window_min = 60        # 1 hour
        window_max = 720       # 12 hours
        window_step = 30
        window_help = f"Rolling window size in minutes for {interval} data stability analysis"
    elif interval == "1h":
        window_unit = "hours"
        window_default = 168   # 1 week in hours
        window_min = 24        # 1 day
        window_max = 720       # 1 month
        window_step = 24
        window_help = "Rolling window size in hours for stability analysis"
    else:
        window_unit = "days"
        window_default = 252   # 1 trading year
        window_min = 50
        window_max = 500
        window_step = 10
        window_help = "Rolling window size in days for stability analysis"
    
    rolling_window_size = st.sidebar.slider(
        f"Rolling Window Size ({window_unit}):",
        min_value=window_min,
        max_value=window_max,
        value=window_default,
        step=window_step,
        disabled=not enable_rolling_analysis,
        help=window_help
    )
    
    apply_multiple_correction = st.sidebar.checkbox(
        "Apply Multiple Comparison Correction",
        value=True,
        help="Apply Benjamini-Hochberg FDR correction for multiple testing"
    )
    
    # Correlation Discovery Parameters (only show for correlation discovery)
    if analysis_type == "Correlation Discovery":
        st.sidebar.subheader("🔍 Discovery-Specific Parameters")
        
        top_n_correlations = st.sidebar.slider(
            "Number of Top Correlations to Find:",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of highest correlated stock pairs to identify"
        )
        
        correlation_min_threshold = st.sidebar.slider(
            "Minimum Correlation for Discovery:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Only show correlations above this threshold"
        )
        # Store in session state for later use
        st.session_state['correlation_min_threshold'] = correlation_min_threshold
        
        st.sidebar.info("💡 This analysis will scan all stock pairs to find the strongest correlations")
    
    # Enhanced Correlation Discovery Parameters (only show for enhanced correlation discovery)
    elif analysis_type == "Enhanced Correlation Discovery":
        st.sidebar.subheader("🚀 Enhanced Discovery Parameters")
        
        st.sidebar.info("🎯 Multi-horizon analysis across different time periods and lag structures")
        
        # Time horizons selection
        st.sidebar.markdown("**📅 Time Horizons to Analyze:**")
        available_periods = ["1w", "1mo", "3mo", "6mo", "1y", "2y"]
        default_horizons = ["1mo", "3mo", "6mo", "1y"]
        
        time_horizons = []
        col1, col2 = st.sidebar.columns(2)
        for i, period_option in enumerate(available_periods):
            with col1 if i % 2 == 0 else col2:
                if st.checkbox(
                    f"{period_option}",
                    value=period_option in default_horizons,
                    key=f"horizon_{period_option}"
                ):
                    time_horizons.append(period_option)
        
        if not time_horizons:
            st.sidebar.warning("⚠️ Please select at least one time horizon")
            time_horizons = ["1y"]  # Default fallback
        
        # Lag periods selection
        st.sidebar.markdown("**⏱️ Lag Periods (days):**")
        lag_periods_input = st.sidebar.text_input(
            "Lag periods (comma-separated):",
            value="1,3,5,10,20",
            help="Enter lag periods in days, separated by commas (e.g., 1,3,5,10,20)"
        )
        
        try:
            lag_periods = [int(x.strip()) for x in lag_periods_input.split(",") if x.strip()]
            if not lag_periods:
                lag_periods = [1, 3, 5, 10, 20]  # Default
        except ValueError:
            st.sidebar.error("Invalid lag periods format. Using default values.")
            lag_periods = [1, 3, 5, 10, 20]
        
        st.sidebar.write(f"Selected lag periods: {', '.join(map(str, lag_periods))} days")
        
        # Top N correlations
        top_n_enhanced = st.sidebar.slider(
            "Top Correlations per Horizon:",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of highest correlated pairs to find for each time horizon"
        )
        
        # Minimum correlation threshold
        correlation_min_threshold_enhanced = st.sidebar.slider(
            "Minimum Correlation Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Only show correlations above this threshold"
        )
        
        # Show analysis scope
        total_combinations = len(time_horizons) * len(lag_periods)
        st.sidebar.success(f"📊 Will analyze {total_combinations} horizon-lag combinations")
        
        # Store enhanced parameters in session state
        st.session_state['time_horizons'] = time_horizons
        st.session_state['lag_periods'] = lag_periods
        st.session_state['top_n_enhanced'] = top_n_enhanced
        st.session_state['correlation_min_threshold_enhanced'] = correlation_min_threshold_enhanced
    
    # Data Transformation Options - context-aware
    st.sidebar.subheader("🔄 Data Preprocessing")
    
    # Show different transformation options based on analysis type
    if analysis_type == "Lead-Lag Analysis":
        st.sidebar.info("🔧 Preprocessing options for lead-lag relationship detection")
        
        enable_detrending = st.sidebar.checkbox(
            "Enable Detrending",
            value=True,  # More important for lead-lag
            help="Remove trends to focus on short-term relationships"
        )
        
        enable_market_adjustment = st.sidebar.checkbox(
            "Remove Market-Wide Effects",
            value=True,  # Very important for lead-lag
            help="Remove market movements to isolate stock-specific relationships"
        )
        
    elif analysis_type in ["Correlation Discovery", "Enhanced Correlation Discovery"]:
        st.sidebar.info("🔧 Preprocessing options for correlation analysis")
        
        enable_detrending = st.sidebar.checkbox(
            "Enable Detrending",
            value=False,  # Less critical for correlation discovery
            help="Remove trends from price data before analysis"
        )
        
        enable_market_adjustment = st.sidebar.checkbox(
            "Remove Market-Wide Effects",
            value=False,  # Optional for correlation discovery
            help="Remove market-wide movements using a benchmark index"
        )
    
    # Common transformation options
    if enable_detrending:
        detrend_method = st.sidebar.selectbox(
            "Detrending Method:",
            options=["linear", "quadratic", "moving_average"],
            index=0,
            help="Method for removing trends from data"
        )
    else:
        detrend_method = "linear"  # Default value
    
    if enable_market_adjustment:
        market_index = st.sidebar.selectbox(
            "Market Benchmark:",
            options=["SPY", "QQQ", "IWM", "VTI", "^GSPC"],
            index=0,
            help="Benchmark index for market adjustment"
        )
        
        market_adjustment_method = st.sidebar.selectbox(
            "Market Adjustment Method:",
            options=["subtract", "beta_adjust", "residual"],
            index=1,
            help="subtract: simple return subtraction, beta_adjust: beta-weighted adjustment, residual: regression residuals"
        )
    else:
        market_index = "SPY"  # Default value
        market_adjustment_method = "beta_adjust"  # Default value
    
    enable_normalization = st.sidebar.checkbox(
        "Enable Data Normalization",
        value=True,
        help="Normalize data for cross-stock comparisons"
    )
    
    if enable_normalization:
        normalization_method = st.sidebar.selectbox(
            "Normalization Method:",
            options=["zscore", "minmax", "robust"],
            index=0,
            help="zscore: standard normalization, minmax: 0-1 scaling, robust: median-based scaling"
        )
    else:
        normalization_method = "zscore"  # Default value
    
    # Context-aware backtest/analysis parameters
    if analysis_type == "Lead-Lag Analysis":
        st.sidebar.subheader("💰 Backtest Parameters")
        st.sidebar.info("📈 Configure backtesting for lead-lag strategies")
        
        backtest_top_n = st.sidebar.slider(
            "Top Strategies to Backtest:", 
            1, 10, 5,
            help="Number of best lead-lag pairs to backtest"
        )
        
        # Trading Cost Parameters for Lead-Lag
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
        
    elif analysis_type in ["Correlation Discovery", "Enhanced Correlation Discovery"]:
        st.sidebar.subheader("📊 Analysis Parameters")
        st.sidebar.info("🔍 Configure parameters for correlation analysis")
        
        # For correlation discovery, show analysis-specific parameters instead of backtest
        confidence_level = st.sidebar.slider(
            "Confidence Level (%):",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="Statistical confidence level for correlation significance"
        )
        
        min_observations = st.sidebar.number_input(
            "Minimum Observations:",
            min_value=30,
            max_value=500,
            value=100,
            step=10,
            help="Minimum number of data points required for analysis"
        )
        
        # Set default values for variables that might be used elsewhere
        backtest_top_n = 5
        enable_trading_costs = False
        transaction_cost = 0.001
        bid_ask_spread = 0.0005
        slippage = 0.0002
    
    # Performance Dashboard
    st.sidebar.subheader("⚡ Performance Dashboard")
    
    # Get performance stats
    perf_stats = performance_monitor.get_performance_summary(last_n_minutes=30)
    
    if "message" not in perf_stats:
        st.sidebar.metric(
            "Avg Fetch Time", 
            f"{perf_stats.get('avg_fetch_time', 0):.2f}s",
            help="Average time to fetch stock data"
        )
        
        st.sidebar.metric(
            "Cache Hit Rate", 
            f"{perf_stats.get('cache_hit_rate', 0):.1f}%",
            help="Percentage of data served from cache"
        )
        
        st.sidebar.metric(
            "Total Operations", 
            perf_stats.get('total_operations', 0),
            help="Number of fetch operations in last 30 minutes"
        )
        
        # Show method usage
        method_usage = perf_stats.get('method_usage', {})
        if method_usage:
            st.sidebar.write("**Fetch Methods Used:**")
            for method, count in method_usage.items():
                st.sidebar.write(f"• {method}: {count}")
    else:
        st.sidebar.info(perf_stats["message"])
    
    # Performance comparison button
    if st.sidebar.button("📊 View Detailed Performance"):
        st.session_state['show_performance'] = True
    
    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    # Main content area
    
    # Show detailed performance if requested
    if st.session_state.get('show_performance', False):
        st.header("⚡ Detailed Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Summary")
            
            # Get comprehensive stats
            full_stats = performance_monitor.get_performance_summary()
            
            if "message" not in full_stats:
                # Create metrics display
                metrics_data = {
                    "Metric": [
                        "Total Operations",
                        "Total Symbols Fetched",
                        "Average Fetch Time",
                        "Median Fetch Time",
                        "Min Fetch Time",
                        "Max Fetch Time",
                        "Cache Hit Rate",
                        "Avg Symbols per Operation",
                        "Operations per Minute"
                    ],
                    "Value": [
                        full_stats.get('total_operations', 0),
                        full_stats.get('total_symbols_fetched', 0),
                        f"{full_stats.get('avg_fetch_time', 0):.3f}s",
                        f"{full_stats.get('median_fetch_time', 0):.3f}s",
                        f"{full_stats.get('min_fetch_time', 0):.3f}s",
                        f"{full_stats.get('max_fetch_time', 0):.3f}s",
                        f"{full_stats.get('cache_hit_rate', 0):.1f}%",
                        f"{full_stats.get('avg_symbols_per_operation', 0):.1f}",
                        f"{full_stats.get('operations_per_minute', 0):.2f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            else:
                st.info(full_stats["message"])
        
        with col2:
            st.subheader("🔄 Method Comparison")
            
            # Get method comparison
            comparison = performance_monitor.get_speed_comparison()
            
            if comparison and isinstance(comparison, dict):
                comparison_data = []
                for method, stats in comparison.items():
                    if isinstance(stats, dict):
                        comparison_data.append({
                            "Method": method,
                            "Operations": stats.get('operations', 0),
                            "Avg Time (s)": f"{stats.get('avg_time', 0):.3f}",
                            "Avg Symbols": f"{stats.get('avg_symbols_per_op', 0):.1f}",
                            "Time per Symbol (s)": f"{stats.get('time_per_symbol', 0):.4f}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create a bar chart for method comparison
                if len(comparison_data) > 1:
                    fig = px.bar(
                        comparison_df, 
                        x="Method", 
                        y="Time per Symbol (s)",
                        title="Fetch Speed by Method (Lower is Better)",
                        color="Method"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No method comparison data available yet.")
        
        # Cache statistics
        st.subheader("💾 Cache Statistics")
        try:
            cache_stats = system.data_fetcher.get_cache_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cache Files", cache_stats['total_files'])
            with col2:
                st.metric("Cache Size", f"{cache_stats['total_size_mb']} MB")
            with col3:
                st.metric("Memory Cache Size", cache_stats['memory_cache_size'])
            with col4:
                hit_rate = cache_stats['memory_cache_hits'] / max(1, cache_stats['memory_cache_hits'] + cache_stats['memory_cache_misses']) * 100
                st.metric("Memory Hit Rate", f"{hit_rate:.1f}%")
        except Exception as e:
            st.warning(f"Could not load cache statistics: {e}")
        
        # Clear performance data button
        if st.button("🗑️ Clear Performance Data"):
            performance_monitor.clear_metrics()
            st.success("Performance data cleared!")
            st.rerun()
        
        # Close performance view button
        if st.button("❌ Close Performance View"):
            st.session_state['show_performance'] = False
            st.rerun()
        
        st.divider()
    
    if run_analysis:
        if len(symbols) < 2:
            st.error("Please enter at least 2 stock symbols.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if analysis_type == "Lead-Lag Analysis":
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
                    st.session_state['analysis_type'] = analysis_type
            
            elif analysis_type == "Correlation Discovery":
                with st.spinner("Running correlation discovery..."):
                    # Update progress
                    status_text.text("Fetching stock data...")
                    progress_bar.progress(20)
                    
                    # Run correlation discovery
                    correlation_config = {
                        'symbols': symbols,
                        'period': period,
                        'interval': interval,
                        'top_n': top_n_correlations,
                        'correlation_method': correlation_method,
                        'return_method': return_method,
                        'min_correlation': correlation_min_threshold,
                        'lag_days': max_lag,  # Fixed parameter name
                        'enable_monte_carlo': enable_monte_carlo,
                        'monte_carlo_iterations': monte_carlo_iterations,
                        'enable_bootstrap': enable_bootstrap,
                        'bootstrap_iterations': bootstrap_iterations,
                        'apply_multiple_correction': apply_multiple_correction,
                        'save_results': False,
                        # Data transformation options
                        'enable_detrending': enable_detrending,
                        'detrend_method': detrend_method,
                        'enable_market_adjustment': enable_market_adjustment,
                        'market_index': market_index,
                        'market_adjustment_method': market_adjustment_method,
                        'enable_normalization': enable_normalization,
                        'normalization_method': normalization_method
                    }
                    
                    results = system.find_highest_correlations(**correlation_config)
                    
                    progress_bar.progress(100)
                    status_text.text("Correlation discovery complete!")
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['symbols'] = symbols
                    st.session_state['analysis_type'] = analysis_type
            
            elif analysis_type == "Enhanced Correlation Discovery":
                with st.spinner("Running enhanced correlation discovery..."):
                    # Update progress
                    status_text.text("Initializing enhanced correlation analysis...")
                    progress_bar.progress(10)
                    
                    # Get enhanced parameters from session state
                    time_horizons = st.session_state.get('time_horizons', ['1y'])
                    lag_periods = st.session_state.get('lag_periods', [1, 3, 5, 10, 20])
                    top_n_enhanced = st.session_state.get('top_n_enhanced', 10)
                    correlation_min_threshold_enhanced = st.session_state.get('correlation_min_threshold_enhanced', 0.3)
                    
                    # Run enhanced correlation discovery
                    enhanced_config = {
                        'symbols': symbols,
                        'time_horizons': time_horizons,
                        'lag_periods': lag_periods,
                        'interval': interval,
                        'top_n': top_n_enhanced,
                        'correlation_method': correlation_method,
                        'return_method': return_method,
                        'min_correlation': correlation_min_threshold_enhanced,
                        'enable_monte_carlo': enable_monte_carlo,
                        'monte_carlo_iterations': monte_carlo_iterations,
                        'enable_bootstrap': enable_bootstrap,
                        'bootstrap_iterations': bootstrap_iterations,
                        'apply_multiple_correction': apply_multiple_correction,
                        'save_results': True,
                        # Data transformation options
                        'enable_detrending': enable_detrending,
                        'detrend_method': detrend_method,
                        'enable_market_adjustment': enable_market_adjustment,
                        'market_index': market_index,
                        'market_adjustment_method': market_adjustment_method,
                        'enable_normalization': enable_normalization,
                        'normalization_method': normalization_method
                    }
                    
                    status_text.text("Running enhanced correlation analysis across time horizons...")
                    progress_bar.progress(50)
                    
                    results = system.enhanced_correlation_discovery(**enhanced_config)
                    
                    progress_bar.progress(100)
                    status_text.text("Enhanced correlation discovery complete!")
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['symbols'] = symbols
                    st.session_state['analysis_type'] = analysis_type
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        symbols = st.session_state['symbols']
        current_analysis_type = st.session_state.get('analysis_type', 'Lead-Lag Analysis')
        
        if current_analysis_type == "Lead-Lag Analysis":
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
        
        elif current_analysis_type == "Correlation Discovery":
            # Correlation Discovery Summary
            st.markdown('<h2 class="sub-header">🔍 Correlation Discovery Results</h2>', unsafe_allow_html=True)
            
            summary = results.get('summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Stock Pairs",
                    summary.get('total_pairs', 0)
                )
            
            with col2:
                st.metric(
                    "High Correlations Found",
                    summary.get('high_correlations_found', 0)
                )
            
            with col3:
                st.metric(
                    "Highest Correlation",
                    f"{summary.get('highest_correlation', 0):.3f}"
                )
            
            with col4:
                st.metric(
                    "Average of Top Correlations",
                    f"{summary.get('average_top_correlation', 0):.3f}"
                )
            
            # Display correlation discovery results
            if 'correlation_results' in results and not results['correlation_results'].empty:
                st.markdown('<h3 class="sub-header">🏆 Top Correlated Stock Pairs</h3>', unsafe_allow_html=True)
                
                correlation_df = results['correlation_results']
                
                # Display columns based on available data
                display_columns = ['stock1', 'stock2', 'correlation', 'p_value', 'strength']
                if 'adjusted_p_value' in correlation_df.columns:
                    display_columns.append('adjusted_p_value')
                if 'monte_carlo_p_value' in correlation_df.columns:
                    display_columns.append('monte_carlo_p_value')
                if 'bootstrap_ci_lower' in correlation_df.columns and 'bootstrap_ci_upper' in correlation_df.columns:
                    display_columns.extend(['bootstrap_ci_lower', 'bootstrap_ci_upper'])
                
                # Format the dataframe for better display
                display_df = correlation_df[display_columns].copy()
                
                # Format p-values for better readability
                for col in ['p_value', 'adjusted_p_value', 'monte_carlo_p_value']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                
                # Format correlation values
                if 'correlation' in display_df.columns:
                    display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                # Format bootstrap confidence intervals
                for col in ['bootstrap_ci_lower', 'bootstrap_ci_upper']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # Statistical significance indicators
                if 'adjusted_p_value' in correlation_df.columns:
                    significant_after_correction = (correlation_df['adjusted_p_value'] < 0.05).sum()
                    st.info(f"📊 {significant_after_correction} correlations remain significant after multiple comparison correction")
                
                if 'monte_carlo_p_value' in correlation_df.columns:
                    monte_carlo_significant = (correlation_df['monte_carlo_p_value'] < 0.05).sum()
                    st.info(f"🎲 {monte_carlo_significant} correlations confirmed significant by Monte Carlo testing")
                
                # Create correlation network visualization
                st.markdown('<h3 class="sub-header">📊 Correlation Network</h3>', unsafe_allow_html=True)
                
                # Create a simple correlation matrix heatmap
                if len(correlation_df) > 0:
                    # Create correlation matrix for heatmap
                    pivot_data = []
                    for _, row in correlation_df.iterrows():
                        pivot_data.append({'stock_1': row['stock1'], 'stock_2': row['stock2'], 'correlation': row['correlation']})
                        pivot_data.append({'stock_1': row['stock2'], 'stock_2': row['stock1'], 'correlation': row['correlation']})
                    
                    if pivot_data:
                        pivot_df = pd.DataFrame(pivot_data)
                        correlation_matrix = pivot_df.pivot_table(index='stock_1', columns='stock_2', values='correlation', fill_value=0)
                        
                        # Add diagonal (self-correlation = 1)
                        for stock in correlation_matrix.index:
                            if stock in correlation_matrix.columns:
                                correlation_matrix.loc[stock, stock] = 1.0
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.index,
                            colorscale='RdBu',
                            zmid=0,
                            text=correlation_matrix.values,
                            texttemplate="%{text:.3f}",
                            textfont={"size": 10},
                            hoverongaps=False
                        ))
                        
                        fig.update_layout(
                            title="Stock Correlation Matrix",
                            xaxis_title="Stock",
                            yaxis_title="Stock",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                # Show message when no correlations found above threshold
                st.markdown('<h3 class="sub-header">🔍 Correlation Discovery Results</h3>', unsafe_allow_html=True)
                # Get correlation threshold from session state or use default
                threshold = st.session_state.get('correlation_min_threshold', 0.3)
                st.warning(
                    f"⚠️ No correlations found above the minimum threshold of {threshold:.2f}. "
                    "Try lowering the minimum correlation threshold in the sidebar or selecting different stocks."
                )
                
                # Show some guidance
                st.info(
                    "💡 **Tips to find correlations:**\n"
                    "- Lower the minimum correlation threshold (try 0.1-0.2)\n"
                    "- Include more stocks from the same sector\n"
                    "- Try a different time period or interval\n"
                    "- Consider stocks that might move together (e.g., tech stocks, oil companies)"
                )
        
        elif current_analysis_type == "Enhanced Correlation Discovery":
            # Enhanced Correlation Discovery Summary
            st.markdown('<h2 class="sub-header">🚀 Enhanced Correlation Discovery Results</h2>', unsafe_allow_html=True)
            
            summary = results.get('summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Time Horizons Analyzed",
                    summary.get('time_horizons_count', 0)
                )
            
            with col2:
                st.metric(
                    "Total Correlations Found",
                    summary.get('total_correlations_found', 0)
                )
            
            with col3:
                st.metric(
                    "Best Overall Correlation",
                    f"{summary.get('best_overall_correlation', 0):.3f}"
                )
            
            with col4:
                st.metric(
                    "Optimal Time Horizon",
                    summary.get('best_horizon', 'N/A')
                )
            
            # Display enhanced correlation results
            if 'correlation_results' in results and not results['correlation_results'].empty:
                st.markdown('<h3 class="sub-header">🏆 Top Enhanced Correlations</h3>', unsafe_allow_html=True)
                
                enhanced_df = results['correlation_results']
                
                # Display columns for enhanced results
                display_columns = ['stock1', 'stock2', 'time_horizon', 'lag_period', 'correlation', 'p_value', 'strength']
                if 'adjusted_p_value' in enhanced_df.columns:
                    display_columns.append('adjusted_p_value')
                if 'monte_carlo_p_value' in enhanced_df.columns:
                    display_columns.append('monte_carlo_p_value')
                if 'bootstrap_ci_lower' in enhanced_df.columns and 'bootstrap_ci_upper' in enhanced_df.columns:
                    display_columns.extend(['bootstrap_ci_lower', 'bootstrap_ci_upper'])
                
                # Format the dataframe for better display
                display_df = enhanced_df[display_columns].copy()
                
                # Format p-values for better readability
                for col in ['p_value', 'adjusted_p_value', 'monte_carlo_p_value']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                
                # Format correlation values
                if 'correlation' in display_df.columns:
                    display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                # Format bootstrap confidence intervals
                for col in ['bootstrap_ci_lower', 'bootstrap_ci_upper']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # Time Horizon Breakdown
                if 'horizon_breakdown' in results:
                    st.markdown('<h3 class="sub-header">📊 Time Horizon Analysis</h3>', unsafe_allow_html=True)
                    
                    horizon_data = results['horizon_breakdown']
                    horizon_summary = []
                    
                    for horizon, data in horizon_data.items():
                        horizon_summary.append({
                            'Time Horizon': horizon,
                            'Top Correlation': f"{data.get('top_correlation', 0):.4f}",
                            'Average Correlation': f"{data.get('avg_correlation', 0):.4f}",
                            'Correlations Found': data.get('correlations_found', 0),
                            'Best Lag Period': f"{data.get('best_lag_period', 0)} days"
                        })
                    
                    horizon_df = pd.DataFrame(horizon_summary)
                    st.dataframe(horizon_df, use_container_width=True)
                    
                    # Visualization of correlations across time horizons
                    fig = go.Figure()
                    
                    horizons = list(horizon_data.keys())
                    top_correlations = [horizon_data[h].get('top_correlation', 0) for h in horizons]
                    avg_correlations = [horizon_data[h].get('avg_correlation', 0) for h in horizons]
                    
                    fig.add_trace(go.Bar(
                        x=horizons,
                        y=top_correlations,
                        name='Top Correlation',
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=horizons,
                        y=avg_correlations,
                        name='Average Correlation',
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title='Correlation Performance Across Time Horizons',
                        xaxis_title='Time Horizon',
                        yaxis_title='Correlation',
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Show message when no enhanced correlations found
                st.markdown('<h3 class="sub-header">🚀 Enhanced Correlation Discovery Results</h3>', unsafe_allow_html=True)
                threshold = st.session_state.get('correlation_min_threshold_enhanced', 0.3)
                st.warning(
                    f"⚠️ No correlations found above the minimum threshold of {threshold:.2f} across the selected time horizons. "
                    "Try lowering the minimum correlation threshold or selecting different stocks."
                )
                
                # Show enhanced guidance
                st.info(
                    "💡 **Tips for Enhanced Correlation Discovery:**\n"
                    "- Lower the minimum correlation threshold (try 0.1-0.2)\n"
                    "- Select more time horizons for broader analysis\n"
                    "- Include stocks from related sectors\n"
                    "- Try different lag periods\n"
                    "- Consider market conditions during selected periods"
                )
        
        # Initialize results_df for all analysis types
        if current_analysis_type == "Enhanced Correlation Discovery":
            results_df = results.get('correlation_results', pd.DataFrame())
        else:
            results_df = results.get('lead_lag_results', pd.DataFrame()) if current_analysis_type == "Lead-Lag Analysis" else results.get('correlation_results', pd.DataFrame())
        
        # Lead-Lag Analysis specific results
        if current_analysis_type == "Lead-Lag Analysis":
            # Top Predictive Pairs
            summary = results['analysis_summary']
            if 'most_predictive_pairs' in summary and summary['most_predictive_pairs']:
                st.markdown('<h3 class="sub-header">🏆 Top Predictive Relationships</h3>', unsafe_allow_html=True)
                
                top_pairs_df = pd.DataFrame(summary['most_predictive_pairs'])
                
                # Get appropriate lag column name
                lag_column_name = f"lag_{get_time_unit_label(interval)}"
                
                # Display columns based on available data
                display_columns = ['leader', 'follower', 'lag_days', 'correlation', 'p_value']
                if 'adjusted_p_value' in top_pairs_df.columns:
                    display_columns.append('adjusted_p_value')
                if 'monte_carlo_p_value' in top_pairs_df.columns:
                    display_columns.append('monte_carlo_p_value')
                if 'bootstrap_ci_lower' in top_pairs_df.columns and 'bootstrap_ci_upper' in top_pairs_df.columns:
                    display_columns.extend(['bootstrap_ci_lower', 'bootstrap_ci_upper'])
                
                # Create display dataframe with renamed lag column
                display_df = top_pairs_df[display_columns].copy()
                display_df = display_df.rename(columns={'lag_days': lag_column_name})
                
                st.dataframe(
                    display_df,
                    use_container_width=True
                )
                
                # Statistical significance indicators
                if 'adjusted_p_value' in top_pairs_df.columns:
                    significant_after_correction = (top_pairs_df['adjusted_p_value'] < 0.05).sum()
                    st.info(f"📊 {significant_after_correction} relationships remain significant after multiple comparison correction")
                
                if 'monte_carlo_p_value' in top_pairs_df.columns:
                    monte_carlo_significant = (top_pairs_df['monte_carlo_p_value'] < 0.05).sum()
                    st.info(f"🎲 {monte_carlo_significant} relationships confirmed significant by Monte Carlo testing")
            
            # Interactive Visualizations (Lead-Lag Analysis only)
            st.markdown('<h2 class="sub-header">📊 Interactive Visualizations</h2>', unsafe_allow_html=True)
            
            if not results_df.empty:
                # Correlation Heatmap
                st.markdown('<h3 class="sub-header">Correlation Heatmap</h3>', unsafe_allow_html=True)
                
                # Get appropriate unit label for selectbox
                lag_unit_label = get_time_unit_label(interval)
                
                lag_options = sorted(results_df['lag_days'].unique())
                selected_lag = st.selectbox(f"Select lag {lag_unit_label} for heatmap:", lag_options)
                
                heatmap_fig = create_correlation_heatmap(results_df, selected_lag, interval)
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
                        
                        lag_profile_fig = create_lag_profile_plot(results_df, selected_leader, selected_follower, interval)
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
                    predictors_df = best_predictors[selected_stock].copy()
                    
                    # Get appropriate lag column name
                    lag_column_name = f"lag_{get_time_unit_label(interval)}"
                    
                    # Rename the column for display
                    display_df = predictors_df[['leader', 'lag_days', 'correlation', 'p_value', 'strength']].copy()
                    display_df = display_df.rename(columns={'lag_days': lag_column_name})
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True
                    )
            
        # Backtest Results
        backtest_results = results.get('backtest_results', {})
        
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