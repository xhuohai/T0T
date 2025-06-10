"""
改进的T0交易器
基于真实T0交易原理：维持日内仓位不变，通过低买高卖获取价差
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ImprovedT0Trader:
    def __init__(self, config):
        """
        初始化改进的T0交易器

        Args:
            config: T0交易配置
        """
        self.config = config
        self.min_trade_portion = config.get("min_trade_portion", 1/8)  # 最小交易比例
        self.max_trade_portion = config.get("max_trade_portion", 1/3)  # 最大交易比例
        self.price_threshold = config.get("price_threshold", 0.01)  # 价格变动阈值1%
        self.transaction_cost_rate = config.get("transaction_cost_rate", 0.0014)  # 交易成本率0.14%

        # T0交易核心参数
        self.base_position = 0  # 基础仓位（日内需要维持的仓位）
        self.current_holdings = 0  # 当前持仓
        self.current_cash = 0  # 当前现金
        self.daily_trades = []  # 当日交易记录
        self.t0_profit = 0  # T0交易累计收益
        self.total_transaction_costs = 0  # 累计交易成本
        
        # 交易状态
        self.trade_state = {
            'last_buy_price': None,
            'last_sell_price': None,
            'daily_high': None,
            'daily_low': None,
            'position_adjusted_today': False
        }

    def set_base_position(self, base_position):
        """
        设置基础仓位（日内需要维持的仓位）
        
        Args:
            base_position: float，基础仓位数量
        """
        self.base_position = base_position
        if self.current_holdings == 0:
            self.current_holdings = base_position

    def detect_t0_signals(self, minute_data):
        """
        检测T0交易信号
        基于价格波动和技术指标
        
        Args:
            minute_data: DataFrame，分钟级数据
            
        Returns:
            DataFrame: 包含T0交易信号的数据
        """
        # 初始化信号列
        minute_data['t0_buy_signal'] = False
        minute_data['t0_sell_signal'] = False
        minute_data['signal_strength'] = 0.0
        minute_data['signal_type'] = ''
        
        # 计算价格变动
        minute_data['price_change'] = minute_data['close'].pct_change()
        minute_data['high_change'] = minute_data['high'].pct_change()
        minute_data['low_change'] = minute_data['low'].pct_change()
        
        # 计算移动平均线
        minute_data['ma5'] = minute_data['close'].rolling(window=5).mean()
        minute_data['ma10'] = minute_data['close'].rolling(window=10).mean()
        
        # 更新日内高低点
        if self.trade_state['daily_high'] is None:
            self.trade_state['daily_high'] = minute_data['high'].max()
            self.trade_state['daily_low'] = minute_data['low'].min()
        else:
            self.trade_state['daily_high'] = max(self.trade_state['daily_high'], minute_data['high'].max())
            self.trade_state['daily_low'] = min(self.trade_state['daily_low'], minute_data['low'].min())
        
        # 遍历数据检测信号
        for i in range(10, len(minute_data)):  # 从第10个数据点开始，确保有足够历史数据
            current_bar = minute_data.iloc[i]
            
            # 计算相对位置
            daily_range = self.trade_state['daily_high'] - self.trade_state['daily_low']
            if daily_range > 0:
                relative_position = (current_bar['close'] - self.trade_state['daily_low']) / daily_range
            else:
                relative_position = 0.5
            
            # T0买入信号：价格相对较低且有反弹迹象
            if self._should_t0_buy(current_bar, minute_data.iloc[i-5:i], relative_position):
                minute_data.loc[minute_data.index[i], 't0_buy_signal'] = True
                minute_data.loc[minute_data.index[i], 'signal_strength'] = self._calculate_buy_strength(current_bar, relative_position)
                minute_data.loc[minute_data.index[i], 'signal_type'] = 't0_buy'
            
            # T0卖出信号：价格相对较高且有回调迹象
            if self._should_t0_sell(current_bar, minute_data.iloc[i-5:i], relative_position):
                minute_data.loc[minute_data.index[i], 't0_sell_signal'] = True
                minute_data.loc[minute_data.index[i], 'signal_strength'] = self._calculate_sell_strength(current_bar, relative_position)
                minute_data.loc[minute_data.index[i], 'signal_type'] = 't0_sell'
        
        return minute_data

    def _should_t0_buy(self, current_bar, recent_data, relative_position):
        """
        判断是否应该T0买入
        
        Args:
            current_bar: 当前K线数据
            recent_data: 最近的K线数据
            relative_position: 相对位置（0-1）
            
        Returns:
            bool: 是否应该买入
        """
        # 条件1：价格在日内相对低位（下半部分）
        if relative_position > 0.6:
            return False
        
        # 条件2：价格下跌后有反弹迹象
        if current_bar['close'] <= recent_data['close'].min() * 1.002:  # 接近近期低点
            if current_bar['close'] > current_bar['low']:  # 当前价格高于最低价，有反弹迹象
                return True
        
        # 条件3：技术指标支持
        if 'ma5' in current_bar and 'ma10' in current_bar:
            if current_bar['close'] < current_bar['ma5'] and current_bar['ma5'] > current_bar['ma10']:
                return True
        
        return False

    def _should_t0_sell(self, current_bar, recent_data, relative_position):
        """
        判断是否应该T0卖出
        
        Args:
            current_bar: 当前K线数据
            recent_data: 最近的K线数据
            relative_position: 相对位置（0-1）
            
        Returns:
            bool: 是否应该卖出
        """
        # 条件1：价格在日内相对高位（上半部分）
        if relative_position < 0.4:
            return False
        
        # 条件2：价格上涨后有回调迹象
        if current_bar['close'] >= recent_data['close'].max() * 0.998:  # 接近近期高点
            if current_bar['close'] < current_bar['high']:  # 当前价格低于最高价，有回调迹象
                return True
        
        # 条件3：技术指标支持
        if 'ma5' in current_bar and 'ma10' in current_bar:
            if current_bar['close'] > current_bar['ma5'] and current_bar['ma5'] < current_bar['ma10']:
                return True
        
        return False

    def _calculate_buy_strength(self, current_bar, relative_position):
        """
        计算买入信号强度
        
        Args:
            current_bar: 当前K线数据
            relative_position: 相对位置
            
        Returns:
            float: 信号强度
        """
        strength = 1.0
        
        # 位置越低，强度越高
        strength += (0.5 - relative_position) * 2  # 最低位置时额外+1
        
        # 成交量放大
        if 'volume' in current_bar and current_bar['volume'] > 0:
            strength += 0.5
        
        return max(0.5, min(3.0, strength))

    def _calculate_sell_strength(self, current_bar, relative_position):
        """
        计算卖出信号强度
        
        Args:
            current_bar: 当前K线数据
            relative_position: 相对位置
            
        Returns:
            float: 信号强度
        """
        strength = 1.0
        
        # 位置越高，强度越高
        strength += (relative_position - 0.5) * 2  # 最高位置时额外+1
        
        # 成交量放大
        if 'volume' in current_bar and current_bar['volume'] > 0:
            strength += 0.5
        
        return max(0.5, min(3.0, strength))

    def execute_t0_trade(self, signal_time, signal_type, price, signal_strength):
        """
        执行T0交易
        
        Args:
            signal_time: datetime，信号时间
            signal_type: str，信号类型，'t0_buy'或't0_sell'
            price: float，交易价格
            signal_strength: float，信号强度
            
        Returns:
            dict: 交易记录，如果不执行交易则返回None
        """
        # 计算交易数量
        base_volume = self.base_position * self.min_trade_portion
        trade_volume = base_volume * signal_strength
        
        # 限制交易数量
        max_volume = self.base_position * self.max_trade_portion
        trade_volume = min(trade_volume, max_volume)
        
        # 检查是否应该执行交易
        if signal_type == 't0_buy':
            # T0买入：增加持仓，但不能超过基础仓位太多
            if self.current_holdings >= self.base_position * 1.5:  # 持仓已经过多
                return None
            
            # 检查价格是否合适
            if self.trade_state['last_sell_price'] is not None:
                if price >= self.trade_state['last_sell_price'] * 0.995:  # 价格没有明显下跌
                    return None
            
            # 执行买入
            cost = trade_volume * price
            transaction_cost = cost * self.transaction_cost_rate
            total_cost = cost + transaction_cost

            self.current_cash -= total_cost
            self.current_holdings += trade_volume
            self.total_transaction_costs += transaction_cost
            self.trade_state['last_buy_price'] = price
            
        elif signal_type == 't0_sell':
            # T0卖出：减少持仓，但不能低于基础仓位太多
            if self.current_holdings <= self.base_position * 0.5:  # 持仓已经过少
                return None
            
            # 检查是否有足够持仓卖出
            if trade_volume > self.current_holdings:
                trade_volume = self.current_holdings - self.base_position
                if trade_volume <= 0:
                    return None
            
            # 检查价格是否合适
            if self.trade_state['last_buy_price'] is not None:
                if price <= self.trade_state['last_buy_price'] * 1.005:  # 价格没有明显上涨
                    return None
            
            # 执行卖出
            revenue = trade_volume * price
            transaction_cost = revenue * self.transaction_cost_rate
            net_revenue = revenue - transaction_cost

            self.current_cash += net_revenue
            self.current_holdings -= trade_volume
            self.total_transaction_costs += transaction_cost
            self.trade_state['last_sell_price'] = price

            # 计算T0收益（扣除交易成本）
            if self.trade_state['last_buy_price'] is not None:
                gross_profit = (price - self.trade_state['last_buy_price']) * trade_volume
                # 买入和卖出都有交易成本
                total_costs = (self.trade_state['last_buy_price'] * trade_volume + revenue) * self.transaction_cost_rate
                net_profit = gross_profit - total_costs
                self.t0_profit += net_profit
        
        # 记录交易
        trade_value = trade_volume * price
        transaction_cost = trade_value * self.transaction_cost_rate

        trade_record = {
            'time': signal_time,
            'type': signal_type.replace('t0_', ''),  # 'buy' or 'sell'
            'price': price,
            'volume': trade_volume,
            'value': trade_value,
            'transaction_cost': transaction_cost,
            'net_value': trade_value - transaction_cost if signal_type == 't0_sell' else -(trade_value + transaction_cost),
            'holdings_after': self.current_holdings,
            'cash_after': self.current_cash,
            'is_t0_trade': True,
            'signal_strength': signal_strength
        }
        
        self.daily_trades.append(trade_record)
        return trade_record

    def force_position_balance(self, current_price, current_time=None, reason="end_of_day"):
        """
        强制平衡仓位到基础仓位

        Args:
            current_price: float，当前价格
            current_time: datetime，当前交易时间
            reason: str，调整原因

        Returns:
            dict: 调整交易记录，如果无需调整则返回None
        """
        position_diff = self.current_holdings - self.base_position
        
        if abs(position_diff) < 0.01:  # 差异很小，无需调整
            return None
        
        if position_diff > 0:
            # 持仓过多，需要卖出
            revenue = position_diff * current_price
            transaction_cost = revenue * self.transaction_cost_rate
            net_revenue = revenue - transaction_cost

            trade_record = {
                'time': current_time if current_time is not None else datetime.now(),
                'type': 'sell',
                'price': current_price,
                'volume': position_diff,
                'value': revenue,
                'transaction_cost': transaction_cost,
                'net_value': net_revenue,
                'holdings_after': self.base_position,
                'cash_after': self.current_cash + net_revenue,
                'is_forced_adjustment': True,
                'adjustment_reason': reason
            }

            self.current_cash += net_revenue
            self.current_holdings = self.base_position
            self.total_transaction_costs += transaction_cost
            
        else:
            # 持仓过少，需要买入
            buy_volume = abs(position_diff)
            cost = buy_volume * current_price
            transaction_cost = cost * self.transaction_cost_rate
            total_cost = cost + transaction_cost

            trade_record = {
                'time': current_time if current_time is not None else datetime.now(),
                'type': 'buy',
                'price': current_price,
                'volume': buy_volume,
                'value': cost,
                'transaction_cost': transaction_cost,
                'net_value': -total_cost,
                'holdings_after': self.base_position,
                'cash_after': self.current_cash - total_cost,
                'is_forced_adjustment': True,
                'adjustment_reason': reason
            }

            self.current_cash -= total_cost
            self.current_holdings = self.base_position
            self.total_transaction_costs += transaction_cost
        
        return trade_record

    def reset_daily_state(self):
        """重置日内状态"""
        self.daily_trades = []
        self.trade_state = {
            'last_buy_price': None,
            'last_sell_price': None,
            'daily_high': None,
            'daily_low': None,
            'position_adjusted_today': False
        }

    def get_daily_performance(self):
        """
        获取当日T0交易表现
        
        Returns:
            dict: 当日表现统计
        """
        if not self.daily_trades:
            return {
                'trades_count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_volume': 0,
                't0_profit': 0
            }
        
        trades_df = pd.DataFrame(self.daily_trades)
        
        return {
            'trades_count': len(trades_df),
            'buy_count': len(trades_df[trades_df['type'] == 'buy']),
            'sell_count': len(trades_df[trades_df['type'] == 'sell']),
            'total_volume': trades_df['volume'].sum(),
            't0_profit': self.t0_profit
        }
