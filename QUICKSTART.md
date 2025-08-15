# 🚀 Quick Start Guide

Get up and running with the Lead-Lag Stock Analysis System in just a few minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading stock data)
- Terminal/Command Prompt access

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies

**Option A: Automatic Installation (Recommended)**
```bash
# Make the install script executable and run it
./install.sh
```

**Option B: Manual Installation**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Create directories
mkdir -p output plots cache
```

### Step 2: Activate Environment
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Run Your First Analysis

**Option A: Interactive Web Interface (Easiest)**
```bash
streamlit run streamlit_app.py
```
Then open your browser to the displayed URL (usually http://localhost:8501)

**Option B: Command Line**
```bash
python main.py
```

**Option C: Example Scripts**
```bash
python example_usage.py
```

## 🎯 Your First Analysis

### Using the Web Interface

1. **Launch the app**: `streamlit run streamlit_app.py`
2. **Select stocks**: Choose from popular stocks or enter your own symbols
3. **Set parameters**: 
   - Time period: Start with "1y" (1 year)
   - Max lag: Try 5-10 days
   - Min correlation: 0.2 for meaningful relationships
4. **Run analysis**: Click "Run Analysis" and wait for results
5. **Explore results**: View correlations, charts, and backtest performance

### Using the Command Line

```python
# Quick example - analyze tech stocks
from main import LeadLagSystem

system = LeadLagSystem()
results = system.run_full_analysis(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    period='1y',
    max_lag=5,
    min_correlation=0.2
)
```

## 📊 Understanding Your Results

### Key Metrics to Look For:

1. **Correlation Coefficient**: 
   - > 0.5: Strong positive relationship
   - 0.3-0.5: Moderate relationship
   - < 0.3: Weak relationship

2. **P-value**: 
   - < 0.05: Statistically significant
   - < 0.01: Highly significant

3. **Lag Days**: 
   - How many days the leader stock moves before the follower
   - Shorter lags are generally more actionable

4. **Backtest Results**:
   - **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
   - **Max Drawdown**: Largest loss period (lower is better)
   - **Total Return**: Overall profit/loss

## 🔍 Example Analyses to Try

### 1. Tech Stock Leadership
```python
# Who leads the tech sector?
symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA', 'NVDA']
```

### 2. Market Sector Rotation
```python
# Do certain sectors predict others?
symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK']
```

### 3. Crypto vs Tech
```python
# Does crypto predict tech stocks?
symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA']
```

### 4. Economic Indicators
```python
# Do bonds predict stocks?
symbols = ['TLT', 'SPY', 'GLD', 'DXY']
```

## 🛠️ Customization Options

### Time Horizons
- **Daily**: `period='1y'`, `interval='1d'`
- **Hourly**: `period='1mo'`, `interval='1h'`
- **Weekly**: `period='5y'`, `interval='1wk'`

### Analysis Parameters
- **max_lag**: Test 1-30 days (start with 5-10)
- **min_correlation**: Filter threshold (0.1-0.5)
- **method**: 'pearson' (linear) or 'spearman' (rank-based)

### Backtest Settings
- **initial_capital**: Starting money ($10,000 default)
- **transaction_cost**: Trading fees (0.1% default)
- **max_position_size**: Risk limit (50% default)

## 📁 Output Files

After running analysis, check these directories:

- **`output/`**: CSV files with detailed results
- **`plots/`**: Charts and visualizations
- **`cache/`**: Downloaded stock data (for faster re-runs)

## 🚨 Common Issues & Solutions

### "No module named 'yfinance'"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### "No data found for symbol"
- Check if stock symbol is correct (e.g., 'AAPL' not 'Apple')
- Try a different time period
- Some symbols may not have enough historical data

### "Streamlit command not found"
```bash
# Install streamlit specifically
pip install streamlit
```

### Analysis takes too long
- Reduce the number of stocks
- Use shorter time periods
- Increase `min_correlation` threshold
- Reduce `max_lag`

## 🎓 Learning Path

### Beginner (Week 1)
1. Run example analyses with default settings
2. Try different stock combinations
3. Learn to interpret correlation and p-values
4. Understand lag relationships

### Intermediate (Week 2-3)
1. Experiment with different time horizons
2. Customize analysis parameters
3. Analyze backtest results
4. Try sector-specific analyses

### Advanced (Month 1+)
1. Modify the source code for custom strategies
2. Implement additional statistical tests
3. Add new visualization types
4. Integrate with live trading APIs (paper trading only!)

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore example_usage.py** for advanced examples
3. **Check out the source code** to understand the algorithms
4. **Join the discussion** - share your findings!

## ⚠️ Important Disclaimers

- **This is for educational purposes only**
- **Not financial advice** - always do your own research
- **Past performance doesn't predict future results**
- **Consider transaction costs and market impact in real trading**
- **Test thoroughly before any real money decisions**

## 🆘 Need Help?

1. Check the error messages carefully
2. Review this guide and README.md
3. Try the example scripts first
4. Start with simple analyses before complex ones
5. Make sure your internet connection is stable

---

**Happy analyzing! 📈**

Remember: The goal is to find statistically significant relationships that might give you an edge in understanding market dynamics. Always combine quantitative analysis with fundamental research and risk management!