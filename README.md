# Lead-Lag Stock Analysis System 📈

A comprehensive Python system for analyzing lead-lag relationships between stocks to identify potential trading opportunities. This system examines how movements in one stock might predict movements in another stock across different time horizons (minutes, hours, days).

## 🎯 Overview

The Lead-Lag Stock Analysis System helps you:

- **Identify Leading Indicators**: Find stocks that consistently move before others
- **Quantify Relationships**: Calculate correlation coefficients with statistical significance
- **Backtest Strategies**: Test trading strategies based on lead-lag relationships
- **Visualize Results**: Generate comprehensive charts and interactive plots
- **Build Confidence Intervals**: Assess the reliability of predictive relationships

## 🚀 Key Features

### 📊 Analysis Capabilities
- **Multi-timeframe Analysis**: Analyze relationships across different time horizons
- **Statistical Significance Testing**: P-values and confidence intervals for all relationships
- **Correlation Methods**: Support for Pearson and Spearman correlations
- **Flexible Data Sources**: Yahoo Finance integration with caching
- **Comprehensive Metrics**: Sharpe ratio, drawdown, win rate, and more

### 💹 Trading Strategy Backtesting
- **Signal Generation**: Automated buy/sell signals based on leader movements
- **Risk Management**: Position sizing and drawdown controls
- **Performance Analytics**: Detailed performance metrics and comparisons
- **Transaction Cost Modeling**: Realistic trading cost simulation

### 📈 Visualization & Reporting
- **Interactive Web Interface**: Streamlit-based GUI
- **Correlation Heatmaps**: Visual representation of relationships
- **Lag Profile Charts**: Correlation vs. lag analysis
- **Backtest Performance Plots**: Portfolio value, drawdown, and signal charts
- **Strategy Comparison**: Side-by-side performance analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or download the project**:
   ```bash
   cd lead:lagstock
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --help
   ```

### Dependencies

The system uses the following key libraries:
- `yfinance`: Stock data fetching
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy`: Statistical analysis
- `matplotlib/seaborn`: Static plotting
- `plotly`: Interactive visualizations
- `streamlit`: Web interface
- `scikit-learn`: Machine learning utilities

## 🎮 Usage

### Command Line Interface

#### Basic Analysis
```bash
# Analyze tech stocks with default settings
python main.py --symbols AAPL MSFT GOOGL META --period 2y --max-lag 10

# Quick analysis with custom parameters
python main.py --symbols TSLA NVDA AMD --interval 1d --correlation-method pearson

# Banking sector analysis
python main.py --symbols JPM BAC WFC C --period 1y --max-lag 5 --min-correlation 0.2
```

#### Advanced Options
```bash
# High-frequency analysis
python main.py --symbols AAPL MSFT --interval 1h --period 1mo --max-lag 24

# Spearman correlation with custom thresholds
python main.py --symbols GOOGL META AMZN --correlation-method spearman --min-correlation 0.15

# Extended backtest analysis
python main.py --symbols XOM CVX COP --backtest-top-n 10 --period 5y
```

### Web Interface

Launch the interactive Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### Web Interface Features:
- **Stock Selection**: Choose from preset groups or enter custom symbols
- **Parameter Configuration**: Adjust analysis parameters via sidebar
- **Real-time Analysis**: Run analysis with progress tracking
- **Interactive Charts**: Zoom, pan, and explore results
- **Data Export**: Download results as CSV files

### Python API

```python
from main import LeadLagSystem

# Initialize system
system = LeadLagSystem()

# Run analysis
results = system.run_full_analysis(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    period='2y',
    max_lag=10,
    correlation_method='pearson'
)

# Access results
print(results['analysis_summary'])
print(results['lead_lag_results'].head())
```

## 📊 Understanding the Results

### Lead-Lag Analysis Output

The system generates several types of results:

#### 1. Correlation Matrix
- **Leader**: Stock that potentially leads
- **Follower**: Stock that potentially follows
- **Lag Days**: Time delay between movements
- **Correlation**: Strength of relationship (-1 to 1)
- **P-value**: Statistical significance
- **Confidence Interval**: Reliability range

#### 2. Best Predictors
For each stock, the system identifies:
- Top leading indicators
- Optimal lag periods
- Strength categories (Strong, Moderate, Weak)
- Statistical significance levels

#### 3. Backtest Results
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return vs. maximum drawdown

### Interpretation Guidelines

#### Strong Relationships (|r| > 0.7)
- High predictive potential
- Consider for primary strategies
- Monitor for regime changes

#### Moderate Relationships (0.5 < |r| < 0.7)
- Good supplementary indicators
- Combine with other signals
- Useful for diversification

#### Weak Relationships (0.3 < |r| < 0.5)
- Limited standalone value
- May work in specific market conditions
- Consider as confirmation signals

## 📁 Project Structure

```
lead:lagstock/
├── main.py                 # Main CLI application
├── streamlit_app.py        # Web interface
├── data_fetcher.py         # Stock data retrieval
├── lead_lag_analyzer.py    # Core analysis engine
├── backtester.py          # Strategy backtesting
├── visualizer.py          # Plotting and charts
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── cache/                # Data cache directory
├── output/               # Analysis results
└── plots/                # Generated visualizations
```

### Core Modules

#### `data_fetcher.py`
- Yahoo Finance integration
- Data caching and validation
- Multi-symbol batch processing
- Rate limiting and error handling

#### `lead_lag_analyzer.py`
- Correlation analysis with confidence intervals
- Statistical significance testing
- Multiple correlation methods
- Lag optimization

#### `backtester.py`
- Signal generation from lead-lag relationships
- Portfolio simulation with transaction costs
- Risk management and position sizing
- Performance metric calculation

#### `visualizer.py`
- Static and interactive plotting
- Correlation heatmaps
- Lag profile analysis
- Backtest performance charts

## ⚙️ Configuration Options

### Data Parameters
- **Period**: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
- **Interval**: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`
- **Symbols**: Any valid stock ticker symbols

### Analysis Parameters
- **Max Lag**: Maximum days to test (1-30)
- **Correlation Method**: `pearson` or `spearman`
- **Return Method**: `pct_change`, `log_returns`, or `diff`
- **Min Correlation**: Threshold for significance (0.0-1.0)

### Backtest Parameters
- **Threshold**: Signal generation threshold (default: 2%)
- **Max Position Size**: Maximum portfolio allocation (default: 10%)
- **Transaction Cost**: Trading cost percentage (default: 0.1%)
- **Risk-Free Rate**: For Sharpe ratio calculation (default: 2%)

## 📈 Example Analysis Workflow

### 1. Hypothesis Formation
```
Hypothesis: "Microsoft stock movements predict Apple movements with a 2-day lag"
```

### 2. Data Collection
```bash
python main.py --symbols MSFT AAPL --period 2y --max-lag 5
```

### 3. Result Analysis
- Check correlation at 2-day lag
- Verify statistical significance (p < 0.05)
- Examine confidence intervals

### 4. Strategy Development
- If correlation > 0.5 and p < 0.05:
  - Buy AAPL when MSFT rises > 2%
  - Hold for optimal period
  - Set stop-loss and take-profit levels

### 5. Backtesting
- Test strategy on historical data
- Analyze risk-adjusted returns
- Compare to buy-and-hold benchmark

### 6. Implementation Considerations
- Transaction costs
- Market impact
- Regime changes
- Position sizing

## 🔧 Advanced Usage

### Custom Analysis Scripts

```python
from data_fetcher import StockDataFetcher
from lead_lag_analyzer import LeadLagAnalyzer

# Custom analysis
fetcher = StockDataFetcher()
analyzer = LeadLagAnalyzer(min_periods=50)

# Get data
prices = fetcher.get_price_data(['AAPL', 'MSFT'], period='1y')

# Analyze specific pair
results = analyzer.analyze_lead_lag_pair(
    prices['MSFT'], prices['AAPL'], 
    'MSFT', 'AAPL', max_lag=10
)

# Print results
for result in results:
    print(f"Lag {result.lag_days}: r={result.correlation:.3f}, p={result.p_value:.4f}")
```

### Batch Processing

```python
# Analyze multiple sectors
sectors = {
    'tech': ['AAPL', 'MSFT', 'GOOGL'],
    'finance': ['JPM', 'BAC', 'WFC'],
    'energy': ['XOM', 'CVX', 'COP']
}

for sector, symbols in sectors.items():
    results = system.run_full_analysis(
        symbols=symbols,
        save_results=True
    )
    print(f"{sector}: {results['analysis_summary']['significant_relationships']} relationships")
```

## 🚨 Important Considerations

### Statistical Warnings
- **Multiple Testing**: With many pairs, some correlations may appear significant by chance
- **Survivorship Bias**: Only analyze currently traded stocks
- **Look-Ahead Bias**: Ensure signals use only past information
- **Regime Changes**: Relationships may break down during market stress

### Trading Risks
- **Transaction Costs**: Can erode profits from frequent trading
- **Market Impact**: Large orders may move prices unfavorably
- **Liquidity**: Some stocks may be difficult to trade quickly
- **Correlation Breakdown**: Relationships may weaken or reverse

### Best Practices
- **Out-of-Sample Testing**: Reserve recent data for validation
- **Rolling Analysis**: Update relationships periodically
- **Risk Management**: Use stop-losses and position limits
- **Diversification**: Don't rely on single relationships

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional data sources (Alpha Vantage, Quandl, etc.)
- More sophisticated signal generation methods
- Machine learning-based relationship detection
- Real-time trading integration
- Enhanced risk management features

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with data provider terms of service and applicable financial regulations.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not intended as financial advice or a recommendation to buy or sell securities. Past performance does not guarantee future results. Trading stocks involves substantial risk of loss and is not suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.**

---

**Happy Analyzing! 📊🚀**