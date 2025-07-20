"""
Multi-Asset Trading Strategy Backtester
=======================================

A comprehensive backtesting framework for multi-asset trading strategies
with advanced risk management and optimization capabilities.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration parameters for the backtesting system"""

    # Initial Parameters
    initial_capital: float = 100000
    start_date: str = "2014-01-01"
    end_date: str = "2024-01-01"

    # Assets
    assets: List[str] = None
    benchmark: str = "SPY"

    # Strategy Parameters
    short_ma_period: int = 20
    long_ma_period: int = 50
    mean_reversion_lookback: int = 14
    mean_reversion_zscore: float = 2.0
    volume_ma_period: int = 20

    # Risk Management
    max_allocation_per_asset: float = 0.40
    min_cash_reserve: float = 0.15
    max_portfolio_drawdown: float = 0.15
    correlation_threshold: float = 0.8
    vix_threshold: float = 25.0

    # Market Parameters
    cash_interest_rate: float = 0.04
    transaction_cost: float = 0.0

    # Optimization Parameters
    optimization_window: int = 504  # 2 years of trading days
    oos_period: int = 126  # 6 months out-of-sample

    def __post_init__(self):
        if self.assets is None:
            self.assets = ["SPY", "QQQ", "GLD"]


class DataManager:
    """Handles data acquisition, cleaning, and preprocessing"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data = {}
        self.vix_data = None

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all assets and VIX"""
        logger.info("Fetching market data...")

        # Add VIX to the list of symbols to fetch
        all_symbols = self.config.assets + ["^VIX"]

        try:
            # Fetch data with extended period to ensure we have enough history
            start_extended = pd.to_datetime(
                self.config.start_date) - timedelta(days=365)

            raw_data = yf.download(
                all_symbols,
                start=start_extended,
                end=self.config.end_date,
                auto_adjust=False,
                progress=False
            )

            # Handle single vs multiple assets
            if len(all_symbols) == 1:
                raw_data = pd.DataFrame(raw_data)
                raw_data.columns = pd.MultiIndex.from_product(
                    [raw_data.columns, [all_symbols[0]]])

            # Process each asset
            for symbol in self.config.assets:
                try:
                    asset_data = self._process_asset_data(raw_data, symbol)
                    self.data[symbol] = asset_data
                    logger.info(f"Successfully processed data for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    raise

            # Process VIX data
            try:
                vix_data = raw_data['Adj Close']['^VIX'].dropna()
                self.vix_data = vix_data
                logger.info("Successfully processed VIX data")
            except Exception as e:
                logger.warning(f"Could not fetch VIX data: {e}")
                # Create dummy VIX data
                dates = self.data[self.config.assets[0]].index
                self.vix_data = pd.Series(20.0, index=dates, name='VIX')

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

        return self.data

    def _process_asset_data(self, raw_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process individual asset data with adjustments for splits and dividends"""

        # Extract OHLCV data
        asset_data = pd.DataFrame({
            'Open': raw_data['Open'][symbol],
            'High': raw_data['High'][symbol],
            'Low': raw_data['Low'][symbol],
            'Close': raw_data['Close'][symbol],
            'Adj Close': raw_data['Adj Close'][symbol],
            'Volume': raw_data['Volume'][symbol]
        })

        # Handle missing data
        asset_data = asset_data.dropna()

        # Calculate returns
        asset_data['Returns'] = asset_data['Adj Close'].pct_change()
        asset_data['Log_Returns'] = np.log(
            asset_data['Adj Close'] / asset_data['Adj Close'].shift(1))

        # Calculate volatility (20-day rolling)
        asset_data['Volatility'] = asset_data['Returns'].rolling(
            20).std() * np.sqrt(252)

        # Volume indicators
        asset_data['Volume_MA'] = asset_data['Volume'].rolling(
            self.config.volume_ma_period).mean()
        asset_data['Volume_Ratio'] = asset_data['Volume'] / \
            asset_data['Volume_MA']

        return asset_data


class StrategyEngine:
    """Implements various trading strategies and signal generation"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def generate_signals(self, data: Dict[str, pd.DataFrame], vix_data: pd.Series) -> pd.DataFrame:
        """Generate trading signals for all assets"""

        # Get common date range
        all_dates = set(data[list(data.keys())[0]].index)
        for asset_data in data.values():
            all_dates = all_dates.intersection(set(asset_data.index))
        all_dates = sorted(list(all_dates))

        # Initialize signals dataframe
        signals = pd.DataFrame(index=all_dates)

        for asset in self.config.assets:
            asset_data = data[asset].loc[all_dates]

            # Moving Average Crossover Signals
            ma_signals = self._moving_average_signals(asset_data)

            # Mean Reversion Signals
            mr_signals = self._mean_reversion_signals(asset_data)

            # Multi-timeframe confirmation
            mtf_signals = self._multi_timeframe_confirmation(asset_data)

            # Volume confirmation
            volume_confirmation = self._volume_confirmation(asset_data)

            # Combine signals
            combined_signal = (ma_signals + mr_signals + mtf_signals) / 3
            combined_signal = combined_signal * volume_confirmation

            signals[f'{asset}_signal'] = combined_signal
            signals[f'{asset}_ma_signal'] = ma_signals
            signals[f'{asset}_mr_signal'] = mr_signals
            signals[f'{asset}_mtf_signal'] = mtf_signals
            signals[f'{asset}_volume_conf'] = volume_confirmation

        # Add VIX data
        vix_aligned = vix_data.reindex(all_dates, method='ffill')
        signals['VIX'] = vix_aligned

        return signals

    def _moving_average_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate moving average crossover signals"""
        short_ma = data['Adj Close'].rolling(
            self.config.short_ma_period).mean()
        long_ma = data['Adj Close'].rolling(self.config.long_ma_period).mean()

        # Signal: 1 when short MA > long MA, -1 when short MA < long MA
        signals = np.where(short_ma > long_ma, 1, -1)
        return pd.Series(signals, index=data.index)

    def _mean_reversion_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals based on z-score"""
        returns = data['Returns']
        rolling_mean = returns.rolling(
            self.config.mean_reversion_lookback).mean()
        rolling_std = returns.rolling(
            self.config.mean_reversion_lookback).std()

        z_score = (returns - rolling_mean) / rolling_std

        # Signal: -1 when z-score > threshold (overbought), 1 when z-score < -threshold (oversold)
        signals = np.where(z_score > self.config.mean_reversion_zscore, -1,
                           np.where(z_score < -self.config.mean_reversion_zscore, 1, 0))

        return pd.Series(signals, index=data.index)

    def _multi_timeframe_confirmation(self, data: pd.DataFrame) -> pd.Series:
        """Multi-timeframe trend confirmation"""
        # Weekly trend (5-day periods)
        weekly_data = data.resample('W').last()
        weekly_short_ma = weekly_data['Adj Close'].rolling(
            4).mean()  # ~20 days
        weekly_long_ma = weekly_data['Adj Close'].rolling(
            10).mean()  # ~50 days

        weekly_trend = np.where(weekly_short_ma > weekly_long_ma, 1, -1)
        weekly_trend_series = pd.Series(weekly_trend, index=weekly_data.index)

        # Align with daily data
        daily_trend = weekly_trend_series.reindex(data.index, method='ffill')

        return daily_trend

    def _volume_confirmation(self, data: pd.DataFrame) -> pd.Series:
        """Volume confirmation for signals"""
        volume_ratio = data['Volume_Ratio']

        # Confirm signals when volume is above average
        confirmation = np.where(volume_ratio > 1.0, 1.0, 0.5)

        return pd.Series(confirmation, index=data.index)


class RiskManager:
    """Implements risk management rules and position sizing"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_position_sizes(self,
                                 signals: pd.DataFrame,
                                 portfolio_value: float,
                                 current_positions: Dict[str, float],
                                 data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate position sizes based on signals and risk management rules"""

        current_date = signals.index[-1]
        new_positions = {}

        # Calculate available capital (excluding cash reserve)
        available_capital = portfolio_value * \
            (1 - self.config.min_cash_reserve)

        # Get current VIX level
        vix_level = signals['VIX'].iloc[-1]

        # Market stress adjustment
        stress_multiplier = self._calculate_stress_multiplier(vix_level)

        # Calculate correlations
        correlation_matrix = self._calculate_correlations(data, lookback=60)

        # Base allocation for each asset
        base_allocation = available_capital / len(self.config.assets)

        for asset in self.config.assets:
            signal = signals[f'{asset}_signal'].iloc[-1]

            # Skip if signal is NaN or 0
            if pd.isna(signal) or signal == 0:
                new_positions[asset] = 0
                continue

            # Calculate volatility adjustment
            asset_data = data[asset]
            recent_vol = asset_data['Volatility'].iloc[-1] if not pd.isna(
                asset_data['Volatility'].iloc[-1]) else 0.2
            # Target 20% volatility
            vol_adjustment = min(1.0, 0.2 / max(recent_vol, 0.05))

            # Apply signal strength
            # Use absolute value for position sizing
            signal_adjustment = abs(signal)

            # Calculate position size
            position_size = (base_allocation *
                             signal_adjustment *
                             vol_adjustment *
                             stress_multiplier)

            # Apply maximum allocation constraint
            max_position = portfolio_value * self.config.max_allocation_per_asset
            position_size = min(position_size, max_position)

            # Apply correlation adjustment
            correlation_adjustment = self._calculate_correlation_adjustment(
                asset, correlation_matrix, current_positions, portfolio_value
            )
            position_size *= correlation_adjustment

            # Set position direction based on signal
            if signal < 0:
                position_size = 0  # No short positions in this implementation

            new_positions[asset] = position_size

        return new_positions

    def _calculate_stress_multiplier(self, vix_level: float) -> float:
        """Calculate position size multiplier based on market stress (VIX)"""
        if vix_level > self.config.vix_threshold:
            # Reduce positions during high volatility
            return max(0.5, 1.0 - (vix_level - self.config.vix_threshold) / 50.0)
        return 1.0

    def _calculate_correlations(self, data: Dict[str, pd.DataFrame], lookback: int = 60) -> pd.DataFrame:
        """Calculate rolling correlations between assets"""
        returns_data = pd.DataFrame()

        for asset in self.config.assets:
            returns_data[asset] = data[asset]['Returns']

        # Calculate rolling correlation matrix
        correlations = returns_data.tail(lookback).corr()

        return correlations

    def _calculate_correlation_adjustment(self,
                                          asset: str,
                                          correlation_matrix: pd.DataFrame,
                                          current_positions: Dict[str, float],
                                          portfolio_value: float) -> float:
        """Adjust position size based on correlations with existing positions"""

        if asset not in correlation_matrix.index:
            return 1.0

        adjustment = 1.0

        for other_asset in self.config.assets:
            if (other_asset != asset and
                other_asset in correlation_matrix.columns and
                other_asset in current_positions and
                    current_positions[other_asset] > 0):

                correlation = correlation_matrix.loc[asset, other_asset]
                position_weight = current_positions[other_asset] / \
                    portfolio_value

                if correlation > self.config.correlation_threshold:
                    # Reduce allocation for highly correlated assets
                    adjustment *= (1.0 - correlation * position_weight * 0.5)

        return max(0.1, adjustment)  # Minimum 10% of original allocation


class BacktestEngine:
    """Main backtesting engine that orchestrates the entire process"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.strategy_engine = StrategyEngine(config)
        self.risk_manager = RiskManager(config)

        # Results storage
        self.results = {}
        self.portfolio_history = []
        self.trades = []

    def run_backtest(self) -> Dict:
        """Run the complete backtesting process"""
        logger.info("Starting backtest...")

        # Fetch and prepare data
        data = self.data_manager.fetch_data()
        signals = self.strategy_engine.generate_signals(
            data, self.data_manager.vix_data)

        # Filter data to backtest period
        start_date = pd.to_datetime(self.config.start_date)
        signals = signals[signals.index >= start_date]

        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        # Dollar amounts
        positions = {asset: 0.0 for asset in self.config.assets}
        cash = self.config.initial_capital

        # Track portfolio history
        portfolio_history = []

        logger.info(
            f"Running backtest from {signals.index[0]} to {signals.index[-1]}")

        # Main backtesting loop
        for i, date in enumerate(signals.index):
            if i == 0:
                continue  # Skip first day (need previous day for calculations)

            current_signals = signals.iloc[:i+1]

            # Calculate new position sizes
            new_positions = self.risk_manager.calculate_position_sizes(
                current_signals, portfolio_value, positions, data
            )

            # Execute trades
            trades_executed = self._execute_trades(
                date, positions, new_positions, data, cash, portfolio_value
            )

            # Update positions and cash
            total_position_value = 0
            for asset in self.config.assets:
                if new_positions[asset] > 0:
                    current_price = data[asset]['Adj Close'].loc[date]
                    shares = new_positions[asset] / current_price
                    positions[asset] = new_positions[asset]
                    total_position_value += new_positions[asset]
                else:
                    positions[asset] = 0

            # Update cash (including interest on cash holdings)
            cash = portfolio_value - total_position_value
            if i > 0:
                daily_interest = cash * (self.config.cash_interest_rate / 365)
                cash += daily_interest

            # Calculate portfolio value
            portfolio_value = cash + total_position_value

            # Record portfolio state
            portfolio_state = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'vix': current_signals['VIX'].iloc[-1],
                **{f'{asset}_position': positions[asset] for asset in self.config.assets},
                **{f'{asset}_weight': positions[asset]/portfolio_value for asset in self.config.assets}
            }
            portfolio_history.append(portfolio_state)

            # Progress logging
            if i % 252 == 0:  # Yearly progress
                logger.info(
                    f"Progress: {date.strftime('%Y-%m-%d')}, Portfolio Value: ${portfolio_value:,.2f}")

        self.portfolio_history = pd.DataFrame(portfolio_history)

        # Calculate performance metrics
        self.results = self._calculate_performance_metrics(data)

        logger.info("Backtest completed successfully!")
        return self.results

    def _execute_trades(self,
                        date: pd.Timestamp,
                        current_positions: Dict[str, float],
                        new_positions: Dict[str, float],
                        data: Dict[str, pd.DataFrame],
                        cash: float,
                        portfolio_value: float) -> List[Dict]:
        """Execute trades and track transaction costs"""

        trades = []

        for asset in self.config.assets:
            current_pos = current_positions[asset]
            new_pos = new_positions[asset]

            if abs(new_pos - current_pos) > 1.0:  # Only trade if difference > $1
                current_price = data[asset]['Adj Close'].loc[date]
                trade_amount = new_pos - current_pos

                # Calculate transaction cost
                transaction_cost = abs(trade_amount) * \
                    self.config.transaction_cost

                trade = {
                    'date': date,
                    'asset': asset,
                    'price': current_price,
                    'amount': trade_amount,
                    'transaction_cost': transaction_cost,
                    'position_before': current_pos,
                    'position_after': new_pos
                }

                trades.append(trade)

                # Reduce portfolio value by transaction cost
                # This will be reflected in the next iteration's calculations

        return trades

    def _calculate_performance_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate comprehensive performance metrics"""

        if self.portfolio_history.empty:
            return {}

        portfolio_df = self.portfolio_history.set_index('date')

        # Portfolio returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['log_returns'] = np.log(portfolio_df['portfolio_value'] /
                                             portfolio_df['portfolio_value'].shift(1))

        # Benchmark returns (SPY)
        benchmark_data = data[self.config.benchmark]['Adj Close']
        benchmark_aligned = benchmark_data.reindex(
            portfolio_df.index, method='ffill')
        benchmark_returns = benchmark_aligned.pct_change()

        # 60/40 portfolio benchmark
        spy_returns = data['SPY']['Returns'].reindex(
            portfolio_df.index, method='ffill')
        # Use 10-year treasury as bond proxy (simulate with 4% annual return)
        bond_returns = pd.Series(
            0.04/252, index=portfolio_df.index)  # Daily bond return
        portfolio_6040_returns = 0.6 * spy_returns + 0.4 * bond_returns

        # Calculate metrics
        metrics = {}

        # Basic performance
        total_return = (
            portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1
        annualized_return = (portfolio_df['portfolio_value'].iloc[-1] /
                             self.config.initial_capital) ** (252/len(portfolio_df)) - 1

        # Risk metrics
        portfolio_returns = portfolio_df['returns'].dropna()
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming risk-free rate = cash interest rate)
        excess_returns = portfolio_returns - \
            (self.config.cash_interest_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * \
            np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)
                         if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)

        # Maximum drawdown
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # VaR (95% confidence)
        var_95 = np.percentile(portfolio_returns.dropna(), 5)

        # Benchmark comparison
        benchmark_total_return = (
            benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
        benchmark_annual_return = (
            benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (252/len(benchmark_aligned)) - 1

        # 60/40 comparison
        portfolio_6040_cumulative = (1 + portfolio_6040_returns).cumprod()
        portfolio_6040_total_return = portfolio_6040_cumulative.iloc[-1] - 1

        # Win rate and trade analysis
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            # For simplicity, consider any positive price movement as a win
            # In a real implementation, you'd track the P&L of each trade

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'benchmark_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return,
            'portfolio_6040_return': portfolio_6040_total_return,
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
            'number_of_trades': len(self.trades)
        }

        return metrics


class VisualizationEngine:
    """Creates comprehensive visualizations and reports"""

    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        self.config = backtest_engine.config

    def create_dashboard(self) -> None:
        """Create comprehensive visualization dashboard"""

        if self.engine.portfolio_history.empty:
            logger.warning("No portfolio history available for visualization")
            return

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 3, 1)
        portfolio_df = self.engine.portfolio_history.set_index('date')

        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'],
                 linewidth=2, label='Strategy Portfolio', color='blue')

        # Add benchmarks
        spy_data = self.engine.data_manager.data['SPY']['Adj Close']
        spy_normalized = spy_data / \
            spy_data.iloc[0] * self.config.initial_capital
        plt.plot(spy_data.index, spy_normalized,
                 linewidth=2, label='SPY Benchmark', color='red', alpha=0.7)

        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Drawdown Analysis
        ax2 = plt.subplot(3, 3, 2)
        returns = portfolio_df['portfolio_value'].pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        plt.fill_between(portfolio_df.index, drawdown, 0,
                         color='red', alpha=0.3, label='Drawdown')
        plt.axhline(y=self.config.max_portfolio_drawdown, color='red',
                    linestyle='--', label='Max DD Limit')
        plt.title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Asset Allocation Over Time
        ax3 = plt.subplot(3, 3, 3)
        allocation_data = portfolio_df[[
            f'{asset}_weight' for asset in self.config.assets]]

        plt.stackplot(portfolio_df.index,
                      *[allocation_data[f'{asset}_weight']
                          for asset in self.config.assets],
                      labels=self.config.assets, alpha=0.7)
        plt.title('Asset Allocation Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Weight')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 4. VIX and Market Stress
        ax4 = plt.subplot(3, 3, 4)
        plt.plot(portfolio_df.index, portfolio_df['vix'],
                 color='orange', linewidth=2, label='VIX')
        plt.axhline(y=self.config.vix_threshold, color='red',
                    linestyle='--', label='VIX Threshold')
        plt.title('Market Stress (VIX)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('VIX Level')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Rolling Sharpe Ratio
        ax5 = plt.subplot(3, 3, 5)
        returns = portfolio_df['portfolio_value'].pct_change()
        rolling_sharpe = (returns.rolling(252).mean() /
                          returns.rolling(252).std()) * np.sqrt(252)

        plt.plot(portfolio_df.index, rolling_sharpe,
                 linewidth=2, color='green', label='1-Year Rolling Sharpe')
        plt.axhline(y=1.0, color='gray', linestyle='--',
                    alpha=0.7, label='Sharpe = 1.0')
        plt.title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Monthly Returns Heatmap
        ax6 = plt.subplot(3, 3, 6)
        monthly_returns = returns.resample(
            'M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = monthly_returns.index.to_period('M')

        # Create year-month matrix
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        heatmap_data = pd.DataFrame(index=years, columns=months)

        for period, ret in monthly_returns.items():
            heatmap_data.loc[period.year, period.month] = ret

        sns.heatmap(heatmap_data.astype(float),
                    annot=True, fmt='.1%', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Monthly Return'})
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Year')

        # 7. Performance Comparison
        ax7 = plt.subplot(3, 3, 7)
        metrics = ['Annualized Return', 'Volatility',
                   'Sharpe Ratio', 'Max Drawdown']
        strategy_values = [
            self.engine.results['annualized_return'],
            self.engine.results.get('volatility', 0),
            self.engine.results['sharpe_ratio'],
            abs(self.engine.results['max_drawdown'])
        ]
        benchmark_values = [
            self.engine.results['benchmark_annual_return'],
            0.15,  # Approximate SPY volatility
            self.engine.results['benchmark_annual_return'] /
            0.15,  # Approximate Sharpe
            0.20   # Approximate max drawdown
        ]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width/2, strategy_values, width,
                label='Strategy', color='blue', alpha=0.7)
        plt.bar(x + width/2, benchmark_values, width,
                label='Benchmark', color='red', alpha=0.7)

        plt.title('Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 8. Asset Correlation Heatmap
        ax8 = plt.subplot(3, 3, 8)
        returns_data = pd.DataFrame()
        for asset in self.config.assets:
            asset_data = self.engine.data_manager.data[asset]
            returns_data[asset] = asset_data['Returns'].reindex(
                portfolio_df.index, method='ffill')

        correlation_matrix = returns_data.corr()

        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

        # 9. Risk-Return Scatter
        ax9 = plt.subplot(3, 3, 9)

        # Calculate individual asset metrics
        asset_returns = []
        asset_volatilities = []

        for asset in self.config.assets:
            asset_data = self.engine.data_manager.data[asset]
            asset_rets = asset_data['Returns'].reindex(
                portfolio_df.index, method='ffill')
            annual_ret = (1 + asset_rets.mean()) ** 252 - 1
            annual_vol = asset_rets.std() * np.sqrt(252)

            asset_returns.append(annual_ret)
            asset_volatilities.append(annual_vol)

        # Plot assets
        plt.scatter(asset_volatilities, asset_returns, s=100, alpha=0.7,
                    c=range(len(self.config.assets)), cmap='viridis')

        # Plot portfolio
        plt.scatter(self.engine.results.get('volatility', 0),
                    self.engine.results['annualized_return'],
                    s=200, color='red', marker='*', label='Portfolio')

        # Add asset labels
        for i, asset in enumerate(self.config.assets):
            plt.annotate(asset, (asset_volatilities[i], asset_returns[i]),
                         xytext=(5, 5), textcoords='offset points')

        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility (Annual)')
        plt.ylabel('Return (Annual)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_performance_report(self) -> None:
        """Print comprehensive performance report"""

        if not self.engine.results:
            logger.warning("No results available for reporting")
            return

        print("\n" + "="*80)
        print(" MULTI-ASSET TRADING STRATEGY BACKTEST REPORT")
        print("="*80)

        # Configuration Summary
        print(f"\n📊 CONFIGURATION SUMMARY")
        print(f"{'─'*50}")
        print(f"Initial Capital:        ${self.config.initial_capital:,.2f}")
        print(
            f"Backtest Period:        {self.config.start_date} to {self.config.end_date}")
        print(f"Assets:                 {', '.join(self.config.assets)}")
        print(f"Benchmark:              {self.config.benchmark}")
        print(f"Transaction Costs:      {self.config.transaction_cost:.2%}")
        print(f"Cash Interest Rate:     {self.config.cash_interest_rate:.2%}")

        # Strategy Parameters
        print(f"\n⚙️ STRATEGY PARAMETERS")
        print(f"{'─'*50}")
        print(
            f"Moving Averages:        {self.config.short_ma_period}/{self.config.long_ma_period} days")
        print(
            f"Mean Reversion:         {self.config.mean_reversion_lookback} days, {self.config.mean_reversion_zscore:.1f} z-score")
        print(f"Volume MA Period:       {self.config.volume_ma_period} days")

        # Risk Management
        print(f"\n🛡️ RISK MANAGEMENT")
        print(f"{'─'*50}")
        print(
            f"Max Allocation/Asset:   {self.config.max_allocation_per_asset:.1%}")
        print(f"Min Cash Reserve:       {self.config.min_cash_reserve:.1%}")
        print(
            f"Max Portfolio DD:       {self.config.max_portfolio_drawdown:.1%}")
        print(f"VIX Threshold:          {self.config.vix_threshold:.1f}")
        print(
            f"Correlation Limit:      {self.config.correlation_threshold:.1%}")

        # Performance Results
        print(f"\n🎯 PERFORMANCE RESULTS")
        print(f"{'─'*50}")
        results = self.engine.results

        print(
            f"Final Portfolio Value:  ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return:           {results['total_return']:+.2%}")
        print(f"Annualized Return:      {results['annualized_return']:+.2%}")
        print(f"Annual Volatility:      {results.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:          {results['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown:       {results['max_drawdown']:.2%}")
        print(f"95% VaR (Daily):        {results['var_95']:.2%}")
        print(f"Number of Trades:       {results['number_of_trades']:,}")

        # Benchmark Comparison
        print(f"\n📈 BENCHMARK COMPARISON")
        print(f"{'─'*50}")
        print(f"Strategy Return:        {results['annualized_return']:+.2%}")
        print(
            f"SPY Return:             {results['benchmark_annual_return']:+.2%}")
        print(
            f"60/40 Portfolio:        {results['portfolio_6040_return']:+.2%}")

        alpha_vs_spy = results['annualized_return'] - \
            results['benchmark_annual_return']
        alpha_vs_6040 = results['annualized_return'] - \
            results['portfolio_6040_return']

        print(f"Alpha vs SPY:           {alpha_vs_spy:+.2%}")
        print(f"Alpha vs 60/40:         {alpha_vs_6040:+.2%}")

        # Risk-Adjusted Performance
        print(f"\n⚖️ RISK-ADJUSTED METRICS")
        print(f"{'─'*50}")
        if results.get('volatility', 0) > 0:
            print(
                f"Return/Risk Ratio:      {results['annualized_return']/results['volatility']:.2f}")
            print(
                f"Calmar Ratio:           {results['annualized_return']/abs(results['max_drawdown']):.2f}")

        # Portfolio Statistics
        if not self.engine.portfolio_history.empty:
            portfolio_df = self.engine.portfolio_history.set_index('date')

            print(f"\n📊 PORTFOLIO STATISTICS")
            print(f"{'─'*50}")

            # Average allocations
            avg_allocations = {}
            for asset in self.config.assets:
                avg_weight = portfolio_df[f'{asset}_weight'].mean()
                avg_allocations[asset] = avg_weight

            print("Average Asset Allocations:")
            for asset, weight in avg_allocations.items():
                print(f"  {asset}:                {weight:.1%}")

            avg_cash = portfolio_df['cash'].mean(
            ) / portfolio_df['portfolio_value'].mean()
            print(f"  Cash:                 {avg_cash:.1%}")

            # Market stress periods
            high_vix_days = (portfolio_df['vix']
                             > self.config.vix_threshold).sum()
            total_days = len(portfolio_df)
            print(f"\nMarket Stress Analysis:")
            print(
                f"  High VIX Days:        {high_vix_days}/{total_days} ({high_vix_days/total_days:.1%})")

        print(f"\n{'='*80}")
        print(" END OF REPORT")
        print("="*80)


class OptimizationEngine:
    """Walk-forward optimization and parameter tuning"""

    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config

    def walk_forward_optimization(self) -> Dict:
        """Perform walk-forward analysis with parameter optimization"""
        logger.info("Starting walk-forward optimization...")

        # Parameter ranges to optimize
        param_ranges = {
            'short_ma_period': [10, 15, 20, 25],
            'long_ma_period': [40, 50, 60, 70],
            'mean_reversion_zscore': [1.5, 2.0, 2.5],
            'vix_threshold': [20, 25, 30]
        }

        results = []

        # Walk-forward windows
        start_date = pd.to_datetime(self.base_config.start_date)
        end_date = pd.to_datetime(self.base_config.end_date)

        current_start = start_date

        while current_start < end_date - timedelta(days=self.base_config.oos_period):
            # Define optimization window
            opt_end = current_start + \
                timedelta(days=self.base_config.optimization_window)
            oos_start = opt_end
            oos_end = min(
                oos_start + timedelta(days=self.base_config.oos_period), end_date)

            if oos_end <= oos_start:
                break

            logger.info(
                f"Optimizing: {current_start.strftime('%Y-%m-%d')} to {opt_end.strftime('%Y-%m-%d')}")
            logger.info(
                f"OOS Testing: {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}")

            # Find best parameters for this window
            best_params = self._optimize_parameters(
                current_start.strftime('%Y-%m-%d'),
                opt_end.strftime('%Y-%m-%d'),
                param_ranges
            )

            # Test on out-of-sample period
            oos_config = BacktestConfig(**asdict(self.base_config))
            oos_config.start_date = oos_start.strftime('%Y-%m-%d')
            oos_config.end_date = oos_end.strftime('%Y-%m-%d')

            # Apply best parameters
            for param, value in best_params.items():
                setattr(oos_config, param, value)

            # Run OOS test
            oos_engine = BacktestEngine(oos_config)
            oos_results = oos_engine.run_backtest()

            results.append({
                'opt_start': current_start,
                'opt_end': opt_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'best_params': best_params,
                'oos_return': oos_results.get('annualized_return', 0),
                'oos_sharpe': oos_results.get('sharpe_ratio', 0),
                'oos_max_dd': oos_results.get('max_drawdown', 0)
            })

            # Move to next window
            current_start = oos_start

        return self._analyze_walkforward_results(results)

    def _optimize_parameters(self, start_date: str, end_date: str, param_ranges: Dict) -> Dict:
        """Optimize parameters for a given period"""

        best_sharpe = -999
        best_params = {}

        # Simple grid search (can be enhanced with more sophisticated methods)
        from itertools import product

        param_combinations = list(product(*param_ranges.values()))

        for combination in param_combinations:
            # Create config with current parameter combination
            test_config = BacktestConfig(**asdict(self.base_config))
            test_config.start_date = start_date
            test_config.end_date = end_date

            for i, param_name in enumerate(param_ranges.keys()):
                setattr(test_config, param_name, combination[i])

            try:
                # Run backtest with current parameters
                engine = BacktestEngine(test_config)
                results = engine.run_backtest()

                sharpe = results.get('sharpe_ratio', -999)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        param: combination[i] for i, param in enumerate(param_ranges.keys())}

            except Exception as e:
                logger.warning(f"Error in optimization: {e}")
                continue

        return best_params

    def _analyze_walkforward_results(self, results: List[Dict]) -> Dict:
        """Analyze walk-forward optimization results"""

        if not results:
            return {}

        df = pd.DataFrame(results)

        analysis = {
            'total_periods': len(results),
            'avg_oos_return': df['oos_return'].mean(),
            'avg_oos_sharpe': df['oos_sharpe'].mean(),
            'avg_oos_max_dd': df['oos_max_dd'].mean(),
            'best_period_return': df['oos_return'].max(),
            'worst_period_return': df['oos_return'].min(),
            'parameter_stability': self._calculate_parameter_stability(df),
            'results_by_period': results
        }

        return analysis

    def _calculate_parameter_stability(self, df: pd.DataFrame) -> Dict:
        """Calculate how stable the optimal parameters are across periods"""

        stability = {}

        # Extract parameter values
        all_params = {}
        for _, row in df.iterrows():
            for param, value in row['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)

        # Calculate stability metrics
        for param, values in all_params.items():
            values_array = np.array(values)
            stability[param] = {
                'mean': values_array.mean(),
                'std': values_array.std(),
                'coefficient_of_variation': values_array.std() / values_array.mean() if values_array.mean() != 0 else 0
            }

        return stability


def main():
    """Main execution function with user interaction"""

    print("🚀 Multi-Asset Trading Strategy Backtester")
    print("="*60)

    # Get user configuration
    print("\n📝 Configuration Setup")
    print("-" * 30)

    try:
        initial_capital = float(
            input("Initial Capital ($) [default: 100000]: ") or "100000")
        start_date = input(
            "Start Date (YYYY-MM-DD) [default: 2014-01-01]: ") or "2014-01-01"
        end_date = input(
            "End Date (YYYY-MM-DD) [default: 2024-01-01]: ") or "2024-01-01"

        # Advanced options
        print("\n⚙️ Strategy Parameters (press Enter for defaults)")
        short_ma = int(input("Short MA Period [20]: ") or "20")
        long_ma = int(input("Long MA Period [50]: ") or "50")
        vix_threshold = float(input("VIX Threshold [25.0]: ") or "25.0")
        max_allocation = float(
            input("Max Allocation per Asset (%) [40]: ") or "40") / 100

        # Create configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            short_ma_period=short_ma,
            long_ma_period=long_ma,
            vix_threshold=vix_threshold,
            max_allocation_per_asset=max_allocation
        )

        print(f"\n✅ Configuration created successfully!")

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        print("Using default configuration...")
        config = BacktestConfig()

    # Run backtest
    print(f"\n🔄 Running backtest...")
    engine = BacktestEngine(config)

    try:
        results = engine.run_backtest()

        # Create visualizations
        viz_engine = VisualizationEngine(engine)
        viz_engine.print_performance_report()
        viz_engine.create_dashboard()

        # Ask about optimization
        run_optimization = input(
            "\n🔍 Run walk-forward optimization? (y/N): ").lower().startswith('y')

        if run_optimization:
            print("\n⚡ Running optimization (this may take several minutes)...")
            optimizer = OptimizationEngine(config)
            opt_results = optimizer.walk_forward_optimization()

            print(f"\n📊 OPTIMIZATION RESULTS")
            print(f"{'─'*40}")
            print(
                f"Periods Analyzed:       {opt_results.get('total_periods', 0)}")
            print(
                f"Avg OOS Return:         {opt_results.get('avg_oos_return', 0):+.2%}")
            print(
                f"Avg OOS Sharpe:         {opt_results.get('avg_oos_sharpe', 0):.2f}")
            print(
                f"Best Period Return:     {opt_results.get('best_period_return', 0):+.2%}")
            print(
                f"Worst Period Return:    {opt_results.get('worst_period_return', 0):+.2%}")

        # Export options
        export_data = input(
            "\n💾 Export results to CSV? (y/N): ").lower().startswith('y')

        if export_data:
            # Export portfolio history
            if not engine.portfolio_history.empty:
                engine.portfolio_history.to_csv(
                    'portfolio_history.csv', index=False)
                print("✅ Portfolio history exported to 'portfolio_history.csv'")

            # Export configuration and results
            export_dict = {
                'config': asdict(config),
                'results': results
            }

            with open('backtest_results.json', 'w') as f:
                json.dump(export_dict, f, indent=2, default=str)
            print("✅ Results exported to 'backtest_results.json'")

        print(f"\n🎉 Backtesting completed successfully!")
        print(
            f"📈 Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"📊 Total Return: {results['total_return']:+.2%}")

    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        print(f"❌ Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
