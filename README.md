# 📈 Enhanced Lead-Lag Stock Analysis System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A comprehensive, institutional-grade system for analyzing lead-lag relationships between stocks using advanced statistical methods, realistic trading costs, and sophisticated data preprocessing.

## 🚀 Live Demo

**[Try the app on Streamlit Community Cloud →](https://your-app-url.streamlit.app)**

## 🎯 Overview

The Lead-Lag Stock Analysis System helps you:

- **Identify Leading Indicators**: Find stocks that consistently move before others
- **Quantify Relationships**: Calculate correlation coefficients with statistical significance
- **Backtest Strategies**: Test trading strategies based on lead-lag relationships
- **Visualize Results**: Generate comprehensive charts and interactive plots
- **Build Confidence Intervals**: Assess the reliability of predictive relationships

## ✨ Key Features

### 🔬 **Advanced Statistical Analysis**
- **Monte Carlo Significance Testing**: Robust statistical validation (100-5000 iterations)
- **Bootstrap Confidence Intervals**: Non-parametric confidence estimation
- **Rolling Window Stability Analysis**: Time-varying correlation assessment
- **Multiple Comparison Correction**: Benjamini-Hochberg FDR control

### 🔄 **Sophisticated Data Processing**
- **Detrending Methods**: Linear, quadratic, and moving average detrending
- **Market Effect Removal**: Subtract, beta-adjust, or residual methods with benchmarks
- **Data Normalization**: Z-score, min-max, and robust scaling options

### 💰 **Realistic Trading Simulation**
- **Transaction Costs**: Configurable commission and fees
- **Bid-Ask Spreads**: Market microstructure modeling
- **Slippage/Market Impact**: Price impact considerations
- **Cost Analysis**: Enable/disable for performance comparison

### 📊 **Professional Visualizations**
- Interactive correlation heatmaps
- Rolling stability analysis plots
- Confidence interval visualizations
- Performance attribution charts

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/lead-lag-stock-analysis.git
cd lead-lag-stock-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run streamlit_app.py
```

### Streamlit Community Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Visit [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account** and select this repository
4. **Deploy** - the app will be live at `https://your-app-name.streamlit.app`

## 📖 Usage Guide

### Web Interface (Recommended)

The Streamlit interface provides an intuitive way to:
1. **Configure Analysis Parameters**: Set stock pairs, date ranges, and analysis methods
2. **Apply Data Transformations**: Choose detrending, market adjustment, and normalization
3. **Set Trading Costs**: Configure realistic transaction costs and market impact
4. **Run Analysis**: Execute comprehensive lead-lag analysis with statistical validation
5. **View Results**: Interactive visualizations and detailed reports

### Programmatic Usage

```python
from main import LeadLagSystem

# Initialize with advanced features
system = LeadLagSystem()

# Run comprehensive analysis
results = system.run_full_analysis(
    stock_pairs=[('AAPL', 'MSFT'), ('GOOGL', 'META')],
    start_date='2020-01-01',
    end_date='2023-12-31',
    # Advanced statistical options
    enable_monte_carlo=True,
    monte_carlo_iterations=1000,
    enable_bootstrap=True,
    bootstrap_iterations=500,
    # Data transformation
    apply_detrending=True,
    detrending_method='linear',
    apply_market_adjustment=True,
    market_index='SPY',
    # Trading costs
    enable_trading_costs=True,
    transaction_cost=0.001,
    bid_ask_spread=0.0005
)
```

## 🔧 Configuration

Key configuration options:

- **Statistical Methods**: Monte Carlo iterations, bootstrap samples, confidence levels
- **Data Processing**: Detrending methods, market benchmarks, normalization techniques
- **Trading Costs**: Commission rates, spreads, slippage parameters
- **Analysis Parameters**: Lag windows, correlation thresholds, significance levels

See `config_example.json` for detailed configuration options.

## 📊 Sample Results

The system provides:
- **Correlation matrices** with statistical significance
- **Lead-lag relationships** with confidence intervals
- **Backtesting results** with realistic costs
- **Stability analysis** over time
- **Performance attribution** and risk metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/lead-lag-stock-analysis/issues) page
2. Create a new issue with detailed description
3. For deployment help, see [Streamlit Documentation](https://docs.streamlit.io/streamlit-community-cloud)

---

**Built with ❤️ using Python, Streamlit, and advanced statistical methods**