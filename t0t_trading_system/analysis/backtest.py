"""
回测模块
用于测试和验证交易策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..strategy.technical.indicators import TechnicalIndicators
from ..strategy.position.position_manager import PositionManager
from ..strategy.t0.t0_trader import T0Trader
from ..risk_management.risk_manager import RiskManager


class Backtest:
    """回测类"""

    def __init__(self, config):
        """
        初始化回测器

        Args:
            config: 回测配置
        """
        self.config = config
        self.start_date = config.get("start_date", "2015-01-01")
        self.end_date = config.get("end_date", "2023-12-31")
        self.initial_capital = config.get("initial_capital", 1000000)
        self.commission_rate = config.get("commission_rate", 0.0003)
        self.slippage = config.get("slippage", 0.0001)
        self.tax_rate = config.get("tax_rate", 0.001)

        # 回测结果
        self.results = {
            'equity_curve': None,
            'trades': None,
            'positions': None,
            'performance_metrics': None
        }

        # 技术指标计算器
        self.indicators = TechnicalIndicators()

    def prepare_data(self, index_data, stock_data):
        """
        准备回测数据

        Args:
            index_data: dict，指数数据，包含月线、周线、日线数据
            stock_data: dict，个股数据，包含日线、分钟线数据

        Returns:
            dict: 处理后的数据
        """
        processed_data = {}

        # 处理指数数据
        if 'monthly' in index_data:
            monthly = index_data['monthly'].copy()
            # 计算月线技术指标
            monthly = self.indicators.calculate_ma(monthly, [5, 10, 20, 60, 144])
            monthly = self.indicators.calculate_macd(monthly)
            monthly = self.indicators.calculate_kdj(monthly)
            # 检测月线背离
            monthly = self.indicators.detect_combined_divergence(monthly)
            processed_data['monthly'] = monthly

        if 'weekly' in index_data:
            weekly = index_data['weekly'].copy()
            # 计算周线技术指标
            weekly = self.indicators.calculate_ma(weekly, [5, 10, 20, 60])
            weekly = self.indicators.calculate_macd(weekly)
            weekly = self.indicators.calculate_kdj(weekly)
            # 检测周线背离
            weekly = self.indicators.detect_combined_divergence(weekly)
            processed_data['weekly'] = weekly

        if 'daily' in index_data:
            daily = index_data['daily'].copy()
            # 计算日线技术指标
            daily = self.indicators.calculate_ma(daily, [5, 10, 20, 60, 120])
            daily = self.indicators.calculate_macd(daily)
            daily = self.indicators.calculate_kdj(daily)
            daily = self.indicators.calculate_atr(daily)
            # 检测日线背离
            daily = self.indicators.detect_combined_divergence(daily)
            processed_data['daily'] = daily

        # 处理个股数据
        if 'daily' in stock_data:
            stock_daily = stock_data['daily'].copy()
            # 计算日线技术指标
            stock_daily = self.indicators.calculate_ma(stock_daily, [5, 10, 20, 60, 120])
            stock_daily = self.indicators.calculate_macd(stock_daily)
            stock_daily = self.indicators.calculate_kdj(stock_daily)
            stock_daily = self.indicators.calculate_atr(stock_daily)
            # 检测日线背离
            stock_daily = self.indicators.detect_combined_divergence(stock_daily)
            processed_data['stock_daily'] = stock_daily

        if 'minute' in stock_data:
            stock_minute = stock_data['minute'].copy()
            # 计算分钟线技术指标
            stock_minute = self.indicators.calculate_macd(stock_minute)
            stock_minute = self.indicators.calculate_kdj(stock_minute)
            # 检测分钟线背离
            stock_minute = self.indicators.detect_combined_divergence(stock_minute, window=10)
            processed_data['stock_minute'] = stock_minute

        return processed_data

    def run_position_strategy(self, data):
        """
        运行仓位管理策略

        Args:
            data: dict，处理后的数据

        Returns:
            DataFrame: 仓位历史记录
        """
        # 初始化仓位管理器
        position_manager = PositionManager(self.config.get("POSITION_CONFIG", {}))

        # 获取月线、周线、日线数据
        monthly_data = data.get('monthly')
        weekly_data = data.get('weekly')
        daily_data = data.get('daily')

        if monthly_data is None or weekly_data is None or daily_data is None:
            raise ValueError("Monthly, weekly and daily data are required for position strategy")

        # 按日期排序
        monthly_data = monthly_data.sort_index()
        weekly_data = weekly_data.sort_index()
        daily_data = daily_data.sort_index()

        # 获取回测日期范围内的日期
        dates = daily_data.loc[
            (daily_data.index >= self.start_date) &
            (daily_data.index <= self.end_date)
        ].index

        # 初始化仓位历史记录
        position_history = []

        # 遍历每个交易日
        for date in tqdm(dates, desc="Running position strategy"):
            # 获取当前日期之前的月线数据
            current_monthly = monthly_data.loc[monthly_data.index <= date]
            if current_monthly.empty:
                continue

            # 获取当前日期之前的周线数据
            current_weekly = weekly_data.loc[weekly_data.index <= date]
            if current_weekly.empty:
                continue

            # 获取当前日期的日线数据
            current_daily = daily_data.loc[daily_data.index == date]
            if current_daily.empty:
                continue

            # 更新仓位
            position = position_manager.update_position(
                current_monthly, current_weekly, current_daily, date)

            # 记录仓位
            position_record = {
                'date': date,
                'position': position
            }
            position_history.append(position_record)

        return pd.DataFrame(position_history)

    def run_t0_strategy(self, data, position_data):
        """
        运行T0交易策略

        Args:
            data: dict，处理后的数据
            position_data: DataFrame，仓位历史记录

        Returns:
            DataFrame: 交易历史记录
        """
        # 初始化T0交易器
        t0_trader = T0Trader(self.config.get("T0_CONFIG", {}))

        # 初始化风险管理器
        risk_manager = RiskManager(self.config.get("RISK_CONFIG", {}))

        # 获取分钟线数据
        minute_data = data.get('stock_minute')
        if minute_data is None:
            raise ValueError("Minute data is required for T0 strategy")

        # 检查分钟数据是否为空
        if minute_data.empty:
            print("Warning: Minute data is empty. Returning empty trade history.")
            return pd.DataFrame(columns=['time', 'type', 'price', 'volume', 'value', 'holdings_after', 'cash_after'])

        # 检查索引是否为日期时间类型
        if not isinstance(minute_data.index, pd.DatetimeIndex):
            print("Warning: Minute data index is not DatetimeIndex. Converting to DatetimeIndex.")
            # 尝试将索引转换为日期时间类型
            try:
                minute_data = minute_data.set_index(pd.DatetimeIndex(minute_data.index))
            except:
                print("Error: Failed to convert minute data index to DatetimeIndex.")
                # 如果转换失败，返回空的交易历史
                return pd.DataFrame(columns=['time', 'type', 'price', 'volume', 'value', 'holdings_after', 'cash_after'])

        # 按日期排序
        minute_data = minute_data.sort_index()

        # 初始化资金和持仓
        t0_trader.current_cash = self.initial_capital
        t0_trader.current_holdings = 0

        # 按日期分组
        try:
            grouped_minute_data = minute_data.groupby(minute_data.index.date)
        except AttributeError:
            print("Warning: Failed to group minute data by date. Using daily data instead.")
            # 如果分组失败，使用日线数据代替
            daily_data = data.get('stock_daily')
            if daily_data is None or daily_data.empty:
                print("Error: Daily data is not available.")
                return pd.DataFrame(columns=['time', 'type', 'price', 'volume', 'value', 'holdings_after', 'cash_after'])

            # 使用日线数据模拟分钟数据
            trade_history = []

            # 遍历每个交易日
            for date, day_position in position_data.iterrows():
                # 获取当日的日线数据
                day_data = daily_data.loc[daily_data.index == date]
                if day_data.empty:
                    continue

                # 获取当日的仓位
                position = day_position['position']

                # 计算当日的目标持仓价值
                target_value = self.initial_capital * position

                # 获取当日开盘价
                open_price = day_data['open'].iloc[0]

                # 计算目标持仓数量
                target_holdings = target_value / open_price

                # 调整持仓到目标水平
                if t0_trader.current_holdings < target_holdings:
                    # 需要买入
                    buy_volume = target_holdings - t0_trader.current_holdings

                    # 执行买入交易
                    trade_record = t0_trader.execute_trade(
                        signal_time=date,
                        signal_type='buy',
                        price=open_price,
                        volume=buy_volume,
                        position_size=1.0  # 这里使用1.0表示买入指定数量
                    )

                    trade_history.append(trade_record)

                elif t0_trader.current_holdings > target_holdings:
                    # 需要卖出
                    sell_volume = t0_trader.current_holdings - target_holdings

                    # 执行卖出交易
                    trade_record = t0_trader.execute_trade(
                        signal_time=date,
                        signal_type='sell',
                        price=open_price,
                        volume=sell_volume,
                        position_size=t0_trader.current_holdings
                    )

                    trade_history.append(trade_record)

            return pd.DataFrame(trade_history)

        # 交易历史记录
        trade_history = []

        # 遍历每个交易日
        for date, day_data in tqdm(grouped_minute_data, desc="Running T0 strategy"):
            # 转换为datetime
            date = pd.Timestamp(date)

            # 获取当日的仓位
            day_position = position_data.loc[position_data['date'].dt.date == date.date()]
            if day_position.empty:
                continue

            position = day_position['position'].iloc[0]

            # 计算当日的目标持仓价值
            target_value = self.initial_capital * position

            # 获取当日开盘价
            open_price = day_data['open'].iloc[0]

            # 计算目标持仓数量
            target_holdings = target_value / open_price

            # 调整持仓到目标水平
            if t0_trader.current_holdings < target_holdings:
                # 需要买入
                buy_volume = target_holdings - t0_trader.current_holdings

                # 执行买入交易
                trade_record = t0_trader.execute_trade(
                    signal_time=day_data.index[0],
                    signal_type='buy',
                    price=open_price,
                    volume=buy_volume,
                    position_size=1.0  # 这里使用1.0表示买入指定数量
                )

                trade_history.append(trade_record)

            elif t0_trader.current_holdings > target_holdings:
                # 需要卖出
                sell_volume = t0_trader.current_holdings - target_holdings

                # 执行卖出交易
                trade_record = t0_trader.execute_trade(
                    signal_time=day_data.index[0],
                    signal_type='sell',
                    price=open_price,
                    volume=sell_volume,
                    position_size=t0_trader.current_holdings
                )

                trade_history.append(trade_record)

            # 检测日内交易信号
            day_data_with_signals = t0_trader.detect_intraday_signals(day_data)

            # 遍历分钟数据执行T0交易
            for i in range(1, len(day_data_with_signals)):
                current_bar = day_data_with_signals.iloc[i]
                current_time = day_data_with_signals.index[i]

                # 检查是否有买入信号
                if current_bar['buy_signal']:
                    # 计算买入数量
                    buy_volume = t0_trader.current_holdings * t0_trader.min_trade_portion
                    if current_bar['signal_strength'] > 1:
                        buy_volume *= current_bar['signal_strength']

                    # 限制买入数量
                    buy_volume = min(buy_volume, t0_trader.current_holdings * t0_trader.max_trade_portion)

                    # 执行买入交易
                    if buy_volume > 0:
                        trade_record = t0_trader.execute_trade(
                            signal_time=current_time,
                            signal_type='buy',
                            price=current_bar['close'],
                            volume=buy_volume,
                            position_size=1.0  # 这里使用1.0表示买入指定数量
                        )

                        trade_history.append(trade_record)

                        # 检查交易是否失败
                        is_failed, failure_reason = risk_manager.check_trade_failure(
                            day_data_with_signals.iloc[i:], trade_record)

                        if is_failed:
                            # 执行止损
                            stop_loss_record = risk_manager.correct_failed_trade(
                                trade_record,
                                day_data_with_signals.iloc[i+1]['open'] if i+1 < len(day_data_with_signals) else current_bar['close'],
                                current_time + timedelta(minutes=1)
                            )

                            # 更新交易记录
                            t0_trader.execute_trade(
                                signal_time=stop_loss_record['time'],
                                signal_type=stop_loss_record['type'],
                                price=stop_loss_record['price'],
                                volume=stop_loss_record['volume'],
                                position_size=t0_trader.current_holdings
                            )

                            trade_history.append(stop_loss_record)

                # 检查是否有卖出信号
                if current_bar['sell_signal']:
                    # 计算卖出数量
                    sell_volume = t0_trader.current_holdings * t0_trader.min_trade_portion
                    if current_bar['signal_strength'] > 1:
                        sell_volume *= current_bar['signal_strength']

                    # 限制卖出数量
                    sell_volume = min(sell_volume, t0_trader.current_holdings * t0_trader.max_trade_portion)

                    # 执行卖出交易
                    if sell_volume > 0 and t0_trader.current_holdings > 0:
                        trade_record = t0_trader.execute_trade(
                            signal_time=current_time,
                            signal_type='sell',
                            price=current_bar['close'],
                            volume=sell_volume,
                            position_size=t0_trader.current_holdings
                        )

                        trade_history.append(trade_record)

                        # 检查交易是否失败
                        is_failed, failure_reason = risk_manager.check_trade_failure(
                            day_data_with_signals.iloc[i:], trade_record)

                        if is_failed:
                            # 执行止损
                            stop_loss_record = risk_manager.correct_failed_trade(
                                trade_record,
                                day_data_with_signals.iloc[i+1]['open'] if i+1 < len(day_data_with_signals) else current_bar['close'],
                                current_time + timedelta(minutes=1)
                            )

                            # 更新交易记录
                            t0_trader.execute_trade(
                                signal_time=stop_loss_record['time'],
                                signal_type=stop_loss_record['type'],
                                price=stop_loss_record['price'],
                                volume=stop_loss_record['volume'],
                                position_size=t0_trader.current_holdings
                            )

                            trade_history.append(stop_loss_record)

        return pd.DataFrame(trade_history)

    def calculate_performance_metrics(self, equity_curve):
        """
        计算绩效指标

        Args:
            equity_curve: DataFrame，权益曲线

        Returns:
            dict: 绩效指标
        """
        # 检查权益曲线是否为空
        if equity_curve is None or equity_curve.empty:
            print("Warning: Equity curve is empty. Returning default performance metrics.")
            return {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'total_trades': 0
            }

        # 检查权益曲线是否至少有两个点
        if len(equity_curve) < 2:
            print("Warning: Equity curve has less than 2 points. Returning default performance metrics.")
            return {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'total_trades': 0
            }

        try:
            # 确保数据按日期排序
            equity_curve = equity_curve.sort_index()

            # 计算日收益率
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()

            # 计算累计收益率
            total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1

            # 计算年化收益率
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

            # 计算最大回撤
            equity_curve['cummax'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['cummax'] - equity_curve['equity']) / equity_curve['cummax']
            max_drawdown = equity_curve['drawdown'].max()

            # 计算夏普比率
            risk_free_rate = 0.03  # 假设无风险利率为3%
            daily_std = equity_curve['daily_return'].std()
            if daily_std > 0:
                sharpe_ratio = (annual_return - risk_free_rate) / (daily_std * np.sqrt(252))
            else:
                sharpe_ratio = 0

            # 计算胜率
            win_trades = len(equity_curve[equity_curve['daily_return'] > 0])
            total_trades = len(equity_curve) - 1  # 减去第一个NaN
            win_rate = win_trades / total_trades if total_trades > 0 else 0

            # 计算盈亏比
            avg_win = equity_curve.loc[equity_curve['daily_return'] > 0, 'daily_return'].mean()
            avg_loss = abs(equity_curve.loc[equity_curve['daily_return'] < 0, 'daily_return'].mean())
            if pd.isna(avg_win):
                avg_win = 0
            if pd.isna(avg_loss) or avg_loss == 0:
                profit_loss_ratio = 0
            else:
                profit_loss_ratio = avg_win / avg_loss

            # 汇总绩效指标
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'total_trades': total_trades
            }

            return metrics
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'total_trades': 0
            }

    def run(self, index_data, stock_data):
        """
        运行回测

        Args:
            index_data: dict，指数数据，包含月线、周线、日线数据
            stock_data: dict，个股数据，包含日线、分钟线数据

        Returns:
            dict: 回测结果
        """
        # 准备数据
        processed_data = self.prepare_data(index_data, stock_data)

        # 运行仓位管理策略
        position_data = self.run_position_strategy(processed_data)

        # 运行T0交易策略
        trade_data = self.run_t0_strategy(processed_data, position_data)

        # 生成权益曲线
        equity_curve = self.generate_equity_curve(trade_data)

        # 计算绩效指标
        performance_metrics = self.calculate_performance_metrics(equity_curve)

        # 保存回测结果
        self.results = {
            'equity_curve': equity_curve,
            'trades': trade_data,
            'positions': position_data,
            'performance_metrics': performance_metrics
        }

        return self.results

    def generate_equity_curve(self, trade_data):
        """
        生成权益曲线

        Args:
            trade_data: DataFrame，交易历史记录

        Returns:
            DataFrame: 权益曲线
        """
        # 检查交易数据是否为空
        if trade_data is None or trade_data.empty:
            print("Warning: Trade data is empty. Returning empty equity curve.")
            # 创建一个包含初始资金的权益曲线
            today = pd.Timestamp.now().date()
            empty_curve = pd.DataFrame({
                'date': [today],
                'equity': [self.initial_capital]
            })
            empty_curve['date'] = pd.to_datetime(empty_curve['date'])
            empty_curve = empty_curve.set_index('date')
            return empty_curve

        # 检查交易数据是否包含必要的列
        required_columns = ['time', 'type', 'value']
        for col in required_columns:
            if col not in trade_data.columns:
                print(f"Warning: Column '{col}' not found in trade_data. Returning empty equity curve.")
                # 创建一个包含初始资金的权益曲线
                today = pd.Timestamp.now().date()
                empty_curve = pd.DataFrame({
                    'date': [today],
                    'equity': [self.initial_capital]
                })
                empty_curve['date'] = pd.to_datetime(empty_curve['date'])
                empty_curve = empty_curve.set_index('date')
                return empty_curve

        try:
            # 确保数据按时间排序
            trade_data = trade_data.sort_values('time')

            # 初始化权益曲线
            equity_data = []

            # 初始权益
            initial_equity = self.initial_capital

            # 第一条记录
            equity_data.append({
                'date': trade_data['time'].iloc[0].date(),
                'equity': initial_equity
            })

            # 遍历每笔交易
            current_equity = initial_equity
            for _, trade in trade_data.iterrows():
                # 更新权益
                if trade['type'] == 'buy':
                    # 买入交易，扣除交易成本
                    cost = trade['value'] * (1 + self.commission_rate + self.slippage)
                    current_equity -= cost
                else:
                    # 卖出交易，增加收益并扣除交易成本
                    revenue = trade['value'] * (1 - self.commission_rate - self.slippage - self.tax_rate)
                    current_equity += revenue

                # 记录权益
                equity_data.append({
                    'date': trade['time'].date(),
                    'equity': current_equity
                })

            # 转换为DataFrame
            equity_curve = pd.DataFrame(equity_data)

            # 设置日期为索引
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])
            equity_curve = equity_curve.set_index('date')

            # 按日期排序
            equity_curve = equity_curve.sort_index()

            # 去除重复日期，保留每日最后一条记录
            equity_curve = equity_curve.groupby(equity_curve.index).last()

            return equity_curve
        except Exception as e:
            print(f"Error generating equity curve: {e}")
            # 创建一个包含初始资金的权益曲线
            today = pd.Timestamp.now().date()
            empty_curve = pd.DataFrame({
                'date': [today],
                'equity': [self.initial_capital]
            })
            empty_curve['date'] = pd.to_datetime(empty_curve['date'])
            empty_curve = empty_curve.set_index('date')
            return empty_curve

    def plot_results(self):
        """
        绘制回测结果

        Returns:
            None
        """
        # 检查回测结果是否存在
        if 'equity_curve' not in self.results or self.results['equity_curve'] is None:
            print("No backtest results to plot")
            return

        # 检查权益曲线是否为空
        equity_curve = self.results['equity_curve']
        if equity_curve.empty:
            print("Equity curve is empty. Nothing to plot.")
            return

        # 检查权益曲线是否至少有两个点
        if len(equity_curve) < 2:
            print("Equity curve has less than 2 points. Not enough data to plot.")
            return

        # 设置绘图风格
        try:
            # 尝试使用新版本的样式名称
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                # 尝试使用旧版本的样式名称
                plt.style.use('seaborn-darkgrid')
            except:
                # 如果都失败，使用默认样式
                pass

        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})

        # 绘制权益曲线
        axes[0].plot(equity_curve.index, equity_curve['equity'], label='Equity Curve')
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity')
        axes[0].legend()

        # 绘制仓位变化
        position_data = self.results['positions']
        if 'date' in position_data.columns and 'position' in position_data.columns and not position_data.empty:
            axes[1].plot(position_data['date'], position_data['position'], label='Position')
            axes[1].set_title('Position Changes')
            axes[1].set_ylabel('Position')
            axes[1].set_ylim(0, 1)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No position data available',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[1].transAxes)
            axes[1].set_title('Position Changes (No Data)')

        # 绘制回撤
        try:
            equity_curve['cummax'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['cummax'] - equity_curve['equity']) / equity_curve['cummax']
            max_drawdown = equity_curve['drawdown'].max()

            axes[2].fill_between(equity_curve.index, equity_curve['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
            axes[2].set_title('Drawdown')
            axes[2].set_ylabel('Drawdown')
            axes[2].set_ylim(0, max(max_drawdown * 1.1, 0.01))  # 确保y轴有一定的范围
            axes[2].legend()
        except Exception as e:
            print(f"Error plotting drawdown: {e}")
            axes[2].text(0.5, 0.5, 'Error plotting drawdown',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[2].transAxes)
            axes[2].set_title('Drawdown (Error)')

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()

        # 打印绩效指标
        print("\nPerformance Metrics:")
        for key, value in self.results['performance_metrics'].items():
            print(f"{key}: {value:.4f}")

    def get_results(self):
        """
        获取回测结果

        Returns:
            dict: 回测结果
        """
        return self.results
