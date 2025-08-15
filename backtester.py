import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    calmar_ratio: float
    
class LeadLagBacktester:
    """Backtests lead-lag trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.02,
                 bid_ask_spread: float = 0.0005,
                 slippage: float = 0.0002,
                 enable_costs: bool = True):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.bid_ask_spread = bid_ask_spread
        self.slippage = slippage
        self.enable_costs = enable_costs
    
    def calculate_signals(self, 
                         leader_data: pd.Series,
                         follower_data: pd.Series,
                         lag_days: int,
                         threshold: float = 0.02,
                         method: str = 'returns') -> pd.Series:
        """Generate trading signals based on leader movements
        
        Args:
            leader_data: Price data for leader stock
            follower_data: Price data for follower stock
            lag_days: Number of days leader leads by
            threshold: Minimum movement threshold to generate signal
            method: 'returns' or 'price_change'
        
        Returns:
            Series with trading signals (1=buy, -1=sell, 0=hold)
        """
        if method == 'returns':
            leader_returns = leader_data.pct_change()
        else:
            leader_returns = leader_data.diff() / leader_data.shift(1)
        
        # Shift leader returns by lag days to create signals
        signals = pd.Series(0, index=follower_data.index)
        
        # Generate signals based on leader movements
        lagged_returns = leader_returns.shift(lag_days)
        
        # Buy signal when leader had strong positive return
        buy_condition = lagged_returns > threshold
        # Sell signal when leader had strong negative return
        sell_condition = lagged_returns < -threshold
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals.fillna(0)
    
    def calculate_position_sizes(self, 
                               signals: pd.Series,
                               prices: pd.Series,
                               portfolio_value: pd.Series,
                               max_position_size: float = 0.1) -> pd.Series:
        """Calculate position sizes based on signals and risk management
        
        Args:
            signals: Trading signals
            prices: Stock prices
            portfolio_value: Current portfolio value
            max_position_size: Maximum position size as fraction of portfolio
        
        Returns:
            Series with position sizes (number of shares)
        """
        positions = pd.Series(0.0, index=signals.index)
        current_position = 0.0
        
        for date in signals.index:
            signal = signals[date]
            price = prices[date]
            portfolio_val = portfolio_value[date]
            
            if pd.isna(price) or pd.isna(portfolio_val) or price <= 0:
                positions[date] = current_position
                continue
            
            if signal == 1:  # Buy signal
                max_shares = (portfolio_val * max_position_size) / price
                target_position = max_shares
            elif signal == -1:  # Sell signal
                target_position = 0.0
            else:  # Hold
                target_position = current_position
            
            positions[date] = target_position
            current_position = target_position
        
        return positions
    
    def simulate_trading(self, 
                        prices: pd.Series,
                        signals: pd.Series,
                        max_position_size: float = 0.1) -> pd.DataFrame:
        """Simulate trading based on signals
        
        Args:
            prices: Stock prices
            signals: Trading signals
            max_position_size: Maximum position size as fraction of portfolio
        
        Returns:
            DataFrame with trading simulation results
        """
        results = pd.DataFrame(index=prices.index)
        results['price'] = prices
        results['signal'] = signals
        
        # Initialize portfolio tracking
        cash = self.initial_capital
        shares = 0.0
        portfolio_values = []
        trade_returns = []
        
        for i, date in enumerate(prices.index):
            price = prices[date]
            signal = signals[date]
            
            if pd.isna(price) or price <= 0:
                portfolio_value = cash + shares * (prices[date-1] if i > 0 else price)
                portfolio_values.append(portfolio_value)
                continue
            
            # Calculate current portfolio value
            portfolio_value = cash + shares * price
            
            # Execute trades based on signals
            if signal == 1 and shares == 0:  # Buy signal and not already holding
                max_investment = portfolio_value * max_position_size
                
                # Calculate effective buy price with costs
                if self.enable_costs:
                    effective_price = price * (1 + self.bid_ask_spread/2 + self.slippage)
                    total_cost_rate = self.transaction_cost
                else:
                    effective_price = price
                    total_cost_rate = 0
                
                shares_to_buy = max_investment / effective_price
                cost = shares_to_buy * effective_price * (1 + total_cost_rate)
                
                if cost <= cash:
                    cash -= cost
                    shares += shares_to_buy
                    
            elif signal == -1 and shares > 0:  # Sell signal and holding shares
                # Calculate effective sell price with costs
                if self.enable_costs:
                    effective_price = price * (1 - self.bid_ask_spread/2 - self.slippage)
                    total_cost_rate = self.transaction_cost
                else:
                    effective_price = price
                    total_cost_rate = 0
                
                proceeds = shares * effective_price * (1 - total_cost_rate)
                
                # Calculate trade return
                entry_price = results['entry_price'].iloc[i-1] if i > 0 else price
                if entry_price > 0:
                    trade_return = (proceeds - (shares * entry_price)) / (shares * entry_price)
                    trade_returns.append(trade_return)
                
                cash += proceeds
                shares = 0.0
            
            # Record entry price for return calculation
            if signal == 1:
                results.loc[date, 'entry_price'] = price
            elif i > 0:
                results.loc[date, 'entry_price'] = results['entry_price'].iloc[i-1]
            
            # Update portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        results['portfolio_value'] = portfolio_values
        results['cash'] = cash
        results['shares'] = shares
        results['returns'] = results['portfolio_value'].pct_change()
        
        return results, trade_returns
    
    def calculate_performance_metrics(self, 
                                    portfolio_values: pd.Series,
                                    trade_returns: List[float],
                                    trading_days_per_year: int = 252) -> BacktestResult:
        """Calculate comprehensive performance metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            trade_returns: List of individual trade returns
            trading_days_per_year: Number of trading days per year
        
        Returns:
            BacktestResult object with all metrics
        """
        # Basic returns
        returns = portfolio_values.pct_change().dropna()
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized metrics
        periods_per_year = trading_days_per_year / len(portfolio_values) * len(returns)
        annualized_return = (1 + total_return) ** (periods_per_year) - 1
        volatility = returns.std() * np.sqrt(trading_days_per_year)
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / trading_days_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(trading_days_per_year) if returns.std() > 0 else 0
        
        # Drawdown analysis
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if trade_returns:
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
            total_trades = len(trade_returns)
        else:
            win_rate = 0
            avg_trade_return = 0
            best_trade = 0
            worst_trade = 0
            total_trades = 0
        
        return BacktestResult(
            strategy_name="Lead-Lag Strategy",
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            calmar_ratio=calmar_ratio
        )
    
    def backtest_strategy(self, 
                         leader_data: pd.Series,
                         follower_data: pd.Series,
                         lag_days: int,
                         threshold: float = 0.02,
                         max_position_size: float = 0.1,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Tuple[BacktestResult, pd.DataFrame]:
        """Run complete backtest for a lead-lag strategy
        
        Args:
            leader_data: Price data for leader stock
            follower_data: Price data for follower stock
            lag_days: Number of days leader leads by
            threshold: Signal threshold
            max_position_size: Maximum position size
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Tuple of (BacktestResult, detailed_results_df)
        """
        # Filter data by date range if specified
        if start_date or end_date:
            mask = pd.Series(True, index=follower_data.index)
            if start_date:
                mask = mask & (follower_data.index >= start_date)
            if end_date:
                mask = mask & (follower_data.index <= end_date)
            
            leader_data = leader_data[mask]
            follower_data = follower_data[mask]
        
        # Generate signals
        signals = self.calculate_signals(leader_data, follower_data, lag_days, threshold)
        
        # Simulate trading
        results_df, trade_returns = self.simulate_trading(follower_data, signals, max_position_size)
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(results_df['portfolio_value'], trade_returns)
        
        return performance, results_df
    
    def compare_strategies(self, 
                         strategies: List[Dict],
                         benchmark_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compare multiple lead-lag strategies
        
        Args:
            strategies: List of strategy configurations
            benchmark_data: Benchmark price data for comparison
        
        Returns:
            DataFrame comparing strategy performance
        """
        results = []
        
        for strategy in strategies:
            try:
                performance, _ = self.backtest_strategy(**strategy)
                
                result_dict = {
                    'strategy': f"{strategy['leader_data'].name} -> {strategy['follower_data'].name} (lag={strategy['lag_days']})",
                    'total_return': performance.total_return,
                    'annualized_return': performance.annualized_return,
                    'volatility': performance.volatility,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'calmar_ratio': performance.calmar_ratio,
                    'win_rate': performance.win_rate,
                    'total_trades': performance.total_trades
                }
                results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Failed to backtest strategy: {e}")
                continue
        
        comparison_df = pd.DataFrame(results)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            benchmark_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1
            benchmark_volatility = benchmark_data.pct_change().std() * np.sqrt(252)
            
            benchmark_row = {
                'strategy': 'Benchmark (Buy & Hold)',
                'total_return': benchmark_return,
                'annualized_return': (1 + benchmark_return) ** (252/len(benchmark_data)) - 1,
                'volatility': benchmark_volatility,
                'sharpe_ratio': (benchmark_return - self.risk_free_rate) / benchmark_volatility,
                'max_drawdown': ((benchmark_data / benchmark_data.expanding().max()) - 1).min(),
                'calmar_ratio': 0,  # Would need more complex calculation
                'win_rate': 0,  # Not applicable for buy & hold
                'total_trades': 1
            }
            
            comparison_df = pd.concat([comparison_df, pd.DataFrame([benchmark_row])], ignore_index=True)
        
        return comparison_df.sort_values('sharpe_ratio', ascending=False)

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate leader stock
    leader_returns = np.random.normal(0.001, 0.02, len(dates))
    leader_prices = pd.Series(100 * np.exp(np.cumsum(leader_returns)), index=dates, name='LEADER')
    
    # Simulate follower stock (correlated with lagged leader)
    follower_returns = np.random.normal(0.0005, 0.015, len(dates))
    # Add some correlation with lagged leader returns
    for i in range(3, len(follower_returns)):
        follower_returns[i] += 0.3 * leader_returns[i-3]
    
    follower_prices = pd.Series(100 * np.exp(np.cumsum(follower_returns)), index=dates, name='FOLLOWER')
    
    # Run backtest
    backtester = LeadLagBacktester()
    performance, results = backtester.backtest_strategy(
        leader_prices, follower_prices, lag_days=3, threshold=0.015
    )
    
    print("Backtest Results:")
    print(f"Total Return: {performance.total_return:.2%}")
    print(f"Annualized Return: {performance.annualized_return:.2%}")
    print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {performance.max_drawdown:.2%}")
    print(f"Win Rate: {performance.win_rate:.2%}")
    print(f"Total Trades: {performance.total_trades}")