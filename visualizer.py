import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LeadLagVisualizer:
    """Creates visualizations for lead-lag analysis results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_correlation_heatmap(self, 
                               results_df: pd.DataFrame, 
                               lag_days: int = 0,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create correlation heatmap for specific lag
        
        Args:
            results_df: Results from lead-lag analysis
            lag_days: Specific lag to visualize
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Filter for specific lag
        lag_data = results_df[results_df['lag_days'] == lag_days]
        
        if lag_data.empty:
            print(f"No data found for lag {lag_days} days")
            return None
        
        # Create pivot table for heatmap
        heatmap_data = lag_data.pivot(index='leader', columns='follower', values='correlation')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title(f'Lead-Lag Correlation Matrix (Lag: {lag_days} days)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Follower Stock', fontsize=12)
        ax.set_ylabel('Leader Stock', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_lag_correlation_profile(self, 
                                   results_df: pd.DataFrame,
                                   leader: str,
                                   follower: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation vs lag for a specific stock pair
        
        Args:
            results_df: Results from lead-lag analysis
            leader: Leader stock symbol
            follower: Follower stock symbol
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Filter for specific pair
        pair_data = results_df[
            (results_df['leader'] == leader) & 
            (results_df['follower'] == follower)
        ].sort_values('lag_days')
        
        if pair_data.empty:
            print(f"No data found for {leader} -> {follower}")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        
        # Plot correlation vs lag
        ax1.plot(pair_data['lag_days'], pair_data['correlation'], 
                'o-', linewidth=2, markersize=6, color=self.colors[0])
        ax1.fill_between(pair_data['lag_days'], 
                        pair_data['ci_lower'], 
                        pair_data['ci_upper'], 
                        alpha=0.3, color=self.colors[0])
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Lag Days')
        ax1.set_ylabel('Correlation')
        ax1.set_title(f'Correlation Profile: {leader} -> {follower}')
        ax1.grid(True, alpha=0.3)
        
        # Plot p-values
        ax2.plot(pair_data['lag_days'], pair_data['p_value'], 
                'o-', linewidth=2, markersize=6, color=self.colors[1])
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.axhline(y=0.01, color='red', linestyle=':', alpha=0.7, label='p=0.01')
        ax2.set_xlabel('Lag Days')
        ax2.set_ylabel('P-value')
        ax2.set_title('Statistical Significance')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_top_predictors(self, 
                          results_df: pd.DataFrame,
                          target_stock: str,
                          top_n: int = 10,
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot top predictors for a target stock
        
        Args:
            results_df: Results from lead-lag analysis
            target_stock: Target stock to find predictors for
            top_n: Number of top predictors to show
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Get top predictors
        predictors = results_df[
            (results_df['follower'] == target_stock) & 
            (results_df['is_significant']) &
            (results_df['lag_days'] > 0)
        ].nlargest(top_n, 'abs_correlation')
        
        if predictors.empty:
            print(f"No significant predictors found for {target_stock}")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create labels with lag information
        labels = [f"{row['leader']} (lag={row['lag_days']})" 
                 for _, row in predictors.iterrows()]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(predictors)), predictors['correlation'], 
                      color=[self.colors[i % len(self.colors)] for i in range(len(predictors))])
        
        # Customize plot
        ax.set_yticks(range(len(predictors)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Correlation')
        ax.set_title(f'Top {top_n} Leading Indicators for {target_stock}', 
                    fontsize=16, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, predictors['correlation'])):
            ax.text(corr + (0.01 if corr > 0 else -0.01), i, f'{corr:.3f}', 
                   va='center', ha='left' if corr > 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_backtest_results(self, 
                            results_df: pd.DataFrame,
                            benchmark_data: Optional[pd.Series] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot backtest results
        
        Args:
            results_df: Backtest results DataFrame
            benchmark_data: Benchmark price data
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        ax1.plot(results_df.index, results_df['portfolio_value'], 
                linewidth=2, color=self.colors[0], label='Strategy')
        
        if benchmark_data is not None:
            # Normalize benchmark to same starting value
            normalized_benchmark = (benchmark_data / benchmark_data.iloc[0]) * results_df['portfolio_value'].iloc[0]
            ax1.plot(benchmark_data.index, normalized_benchmark, 
                    linewidth=2, color=self.colors[1], label='Benchmark', alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        portfolio_values = results_df['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        ax2.fill_between(results_df.index, drawdown, 0, 
                        color=self.colors[3], alpha=0.7)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        returns = results_df['returns'].dropna()
        ax3.hist(returns, bins=50, alpha=0.7, color=self.colors[2], edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Trading signals
        signals = results_df['signal']
        prices = results_df['price']
        
        ax4.plot(results_df.index, prices, linewidth=1, color='black', alpha=0.7)
        
        # Mark buy/sell signals
        buy_signals = results_df[signals == 1]
        sell_signals = results_df[signals == -1]
        
        if not buy_signals.empty:
            ax4.scatter(buy_signals.index, buy_signals['price'], 
                       color='green', marker='^', s=50, label='Buy', zorder=5)
        
        if not sell_signals.empty:
            ax4.scatter(sell_signals.index, sell_signals['price'], 
                       color='red', marker='v', s=50, label='Sell', zorder=5)
        
        ax4.set_title('Trading Signals')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_correlation_plot(self, results_df: pd.DataFrame) -> go.Figure:
        """Create interactive correlation plot using Plotly
        
        Args:
            results_df: Results from lead-lag analysis
        
        Returns:
            Plotly figure
        """
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=results_df['lag_days'],
            y=results_df.index,
            z=results_df['correlation'],
            mode='markers',
            marker=dict(
                size=5,
                color=results_df['correlation'],
                colorscale='RdBu',
                colorbar=dict(title="Correlation"),
                cmin=-1,
                cmax=1
            ),
            text=[f"Leader: {row['leader']}<br>Follower: {row['follower']}<br>Lag: {row['lag_days']}<br>Correlation: {row['correlation']:.3f}<br>P-value: {row['p_value']:.4f}" 
                  for _, row in results_df.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Lead-Lag Correlation Analysis',
            scene=dict(
                xaxis_title='Lag Days',
                yaxis_title='Stock Pair Index',
                zaxis_title='Correlation'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_strategy_comparison(self, 
                               comparison_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of multiple strategies
        
        Args:
            comparison_df: DataFrame with strategy comparison results
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sort by Sharpe ratio for consistent ordering
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=True)
        
        # Sharpe Ratio
        bars1 = ax1.barh(range(len(comparison_df)), comparison_df['sharpe_ratio'],
                         color=[self.colors[i % len(self.colors)] for i in range(len(comparison_df))])
        ax1.set_yticks(range(len(comparison_df)))
        ax1.set_yticklabels(comparison_df['strategy'], fontsize=8)
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio Comparison')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Total Return
        bars2 = ax2.barh(range(len(comparison_df)), comparison_df['total_return'],
                         color=[self.colors[i % len(self.colors)] for i in range(len(comparison_df))])
        ax2.set_yticks(range(len(comparison_df)))
        ax2.set_yticklabels(comparison_df['strategy'], fontsize=8)
        ax2.set_xlabel('Total Return')
        ax2.set_title('Total Return Comparison')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Max Drawdown
        bars3 = ax3.barh(range(len(comparison_df)), comparison_df['max_drawdown'],
                         color=[self.colors[i % len(self.colors)] for i in range(len(comparison_df))])
        ax3.set_yticks(range(len(comparison_df)))
        ax3.set_yticklabels(comparison_df['strategy'], fontsize=8)
        ax3.set_xlabel('Max Drawdown')
        ax3.set_title('Max Drawdown Comparison')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Win Rate
        bars4 = ax4.barh(range(len(comparison_df)), comparison_df['win_rate'],
                         color=[self.colors[i % len(self.colors)] for i in range(len(comparison_df))])
        ax4.set_yticks(range(len(comparison_df)))
        ax4.set_yticklabels(comparison_df['strategy'], fontsize=8)
        ax4.set_xlabel('Win Rate')
        ax4.set_title('Win Rate Comparison')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_plots(self, 
                      results_df: pd.DataFrame,
                      backtest_results: Optional[pd.DataFrame] = None,
                      output_dir: str = "plots") -> None:
        """Save all plots to specified directory
        
        Args:
            results_df: Lead-lag analysis results
            backtest_results: Backtest results (optional)
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation heatmap for lag 0
        self.plot_correlation_heatmap(results_df, lag_days=0, 
                                    save_path=f"{output_dir}/correlation_heatmap_lag0.png")
        
        # Top predictors for each stock
        stocks = results_df['follower'].unique()
        for stock in stocks:
            self.plot_top_predictors(results_df, stock, 
                                   save_path=f"{output_dir}/top_predictors_{stock}.png")
        
        # Backtest results if provided
        if backtest_results is not None:
            self.plot_backtest_results(backtest_results, 
                                     save_path=f"{output_dir}/backtest_results.png")
        
        print(f"All plots saved to {output_dir}/")

# Example usage
if __name__ == "__main__":
    # This would typically be used with real analysis results
    visualizer = LeadLagVisualizer()
    
    # Create sample results data
    sample_results = pd.DataFrame({
        'leader': ['AAPL', 'MSFT', 'GOOGL'] * 3,
        'follower': ['MSFT', 'GOOGL', 'AAPL'] * 3,
        'lag_days': [0, 1, 2] * 3,
        'correlation': [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8, 0.1, 0.9],
        'p_value': [0.01, 0.05, 0.001, 0.02, 0.001, 0.1, 0.001, 0.2, 0.001],
        'ci_lower': [0.3, 0.1, 0.5, 0.2, 0.4, 0.0, 0.6, -0.1, 0.7],
        'ci_upper': [0.7, 0.5, 0.9, 0.6, 0.8, 0.4, 1.0, 0.3, 1.0],
        'is_significant': [True, True, True, True, True, False, True, False, True],
        'abs_correlation': [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8, 0.1, 0.9]
    })
    
    # Create sample plots
    fig1 = visualizer.plot_correlation_heatmap(sample_results, lag_days=0)
    fig2 = visualizer.plot_top_predictors(sample_results, 'AAPL')
    
    plt.show()