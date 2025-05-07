"""
T0交易模块
实现日内高低点确认和T0操作逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class T0Trader:
    """T0交易类"""

    def __init__(self, config):
        """
        初始化T0交易器

        Args:
            config: T0交易配置
        """
        self.config = config
        self.min_trade_portion = config.get("min_trade_portion", 1/8)
        self.max_trade_portion = config.get("max_trade_portion", 1/3)
        self.fib_tolerance = config.get("fib_tolerance", 0.005)
        self.fib_levels = config.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.382, 1.618])
        self.price_tolerance = config.get("price_tolerance", 0.005)

        # 交易记录
        self.trade_history = []

        # 当前持仓
        self.current_holdings = 0

        # 当前现金
        self.current_cash = 0

        # 交易状态
        self.trade_state = {
            'last_trade_time': None,
            'last_trade_price': None,
            'last_trade_type': None,  # 'buy' or 'sell'
            'last_trade_volume': 0,
            'open_trades': [],  # 未平仓的交易
            'pending_signals': []  # 待确认的信号
        }

    def calculate_fibonacci_levels(self, high_price, low_price):
        """
        计算斐波那契回调水平

        Args:
            high_price: float，最高价
            low_price: float，最低价

        Returns:
            dict: 斐波那契水平
        """
        price_range = high_price - low_price

        levels = {}
        for level in self.fib_levels:
            if level <= 1.0:
                levels[str(level)] = low_price + level * price_range
            else:
                levels[str(level)] = high_price + (level - 1.0) * price_range

        return levels

    def is_near_fibonacci_level(self, price, high_price, low_price):
        """
        检查价格是否接近斐波那契水平

        Args:
            price: float，当前价格
            high_price: float，最高价
            low_price: float，最低价

        Returns:
            bool: 是否接近斐波那契水平
            str: 最接近的斐波那契水平
        """
        fib_levels = self.calculate_fibonacci_levels(high_price, low_price)

        min_distance = float('inf')
        nearest_level = None

        for level, level_price in fib_levels.items():
            distance = abs(price - level_price) / price
            if distance < min_distance:
                min_distance = distance
                nearest_level = level

        is_near = min_distance <= self.fib_tolerance

        return is_near, nearest_level

    def detect_intraday_signals(self, minute_data, timeframe='1min'):
        """
        检测日内交易信号

        Args:
            minute_data: DataFrame，分钟级数据，包含技术指标
            timeframe: str，时间框架，如'1min', '5min'等

        Returns:
            DataFrame: 包含交易信号的DataFrame
        """
        # 确保数据按时间排序
        minute_data = minute_data.sort_index()

        # 初始化信号列
        minute_data['buy_signal'] = False
        minute_data['sell_signal'] = False
        minute_data['signal_strength'] = 0  # 信号强度

        # 检查是否包含必要的指标
        required_columns = ['bullish_divergence', 'bearish_divergence', 'macd_dif', 'macd_dea', 'kdj_j', 'kdj_k', 'kdj_d']
        for col in required_columns:
            if col not in minute_data.columns:
                raise ValueError(f"Minute data must contain {col}")

        # 获取昨日K线数据
        yesterday = minute_data.index[0].date() - timedelta(days=1)
        yesterday_data = minute_data.loc[minute_data.index.date == yesterday]

        if not yesterday_data.empty:
            yesterday_high = yesterday_data['high'].max()
            yesterday_low = yesterday_data['low'].min()
        else:
            # 如果没有昨日数据，使用当日的高低点
            today_data = minute_data.loc[minute_data.index.date == minute_data.index[0].date()]
            yesterday_high = today_data['high'].max()
            yesterday_low = today_data['low'].min()

        # 计算可能的斐波那契支撑位和阻力位
        fib_levels = self.calculate_fibonacci_levels(yesterday_high, yesterday_low)

        # 遍历分钟数据检测信号
        for i in range(1, len(minute_data)):
            current_bar = minute_data.iloc[i]
            prev_bar = minute_data.iloc[i-1]

            # 检测底背离买入信号
            if current_bar['bullish_divergence']:
                # 检查是否接近斐波那契水平
                is_near_fib, nearest_level = self.is_near_fibonacci_level(
                    current_bar['low'], yesterday_high, yesterday_low)

                # 检查MACD和KDJ是否同时出现底背离
                macd_bullish = current_bar['macd_dif'] > current_bar['macd_dea']
                kdj_bullish = current_bar['kdj_j'] > current_bar['kdj_d']

                # 设置买入信号
                minute_data.loc[minute_data.index[i], 'buy_signal'] = True

                # 设置信号强度
                signal_strength = 1
                if is_near_fib:
                    signal_strength += 1
                if macd_bullish and kdj_bullish:
                    signal_strength += 1

                minute_data.loc[minute_data.index[i], 'signal_strength'] = signal_strength

            # 检测顶背离卖出信号
            if current_bar['bearish_divergence']:
                # 检查是否接近斐波那契水平
                is_near_fib, nearest_level = self.is_near_fibonacci_level(
                    current_bar['high'], yesterday_high, yesterday_low)

                # 检查MACD和KDJ是否同时出现顶背离
                macd_bearish = current_bar['macd_dif'] < current_bar['macd_dea']
                kdj_bearish = current_bar['kdj_j'] < current_bar['kdj_d']

                # 设置卖出信号
                minute_data.loc[minute_data.index[i], 'sell_signal'] = True

                # 设置信号强度
                signal_strength = 1
                if is_near_fib:
                    signal_strength += 1
                if macd_bearish and kdj_bearish:
                    signal_strength += 1

                minute_data.loc[minute_data.index[i], 'signal_strength'] = signal_strength

        return minute_data

    def execute_trade(self, signal_time, signal_type, price, volume, position_size):
        """
        执行交易

        Args:
            signal_time: datetime，信号时间
            signal_type: str，信号类型，'buy'或'sell'
            price: float，交易价格
            volume: float，交易数量比例（相对于总持仓）
            position_size: float，当前持仓规模

        Returns:
            dict: 交易记录
        """
        # 计算交易数量
        trade_volume = position_size * volume

        # 限制交易数量在允许范围内
        min_volume = position_size * self.min_trade_portion
        max_volume = position_size * self.max_trade_portion

        trade_volume = max(min_volume, min(max_volume, trade_volume))

        # 执行交易
        if signal_type == 'buy':
            # 买入
            cost = trade_volume * price
            self.current_cash -= cost
            self.current_holdings += trade_volume
        else:
            # 卖出
            revenue = trade_volume * price
            self.current_cash += revenue
            self.current_holdings -= trade_volume

        # 记录交易
        trade_record = {
            'time': signal_time,
            'type': signal_type,
            'price': price,
            'volume': trade_volume,
            'value': trade_volume * price,
            'holdings_after': self.current_holdings,
            'cash_after': self.current_cash
        }

        self.trade_history.append(trade_record)

        # 更新交易状态
        self.trade_state['last_trade_time'] = signal_time
        self.trade_state['last_trade_price'] = price
        self.trade_state['last_trade_type'] = signal_type
        self.trade_state['last_trade_volume'] = trade_volume

        # 添加到未平仓交易
        if signal_type == 'buy':
            self.trade_state['open_trades'].append({
                'time': signal_time,
                'price': price,
                'volume': trade_volume
            })
        else:
            # 卖出时，尝试平仓最早的买入交易
            remaining_volume = trade_volume
            while remaining_volume > 0 and self.trade_state['open_trades']:
                oldest_trade = self.trade_state['open_trades'][0]
                if oldest_trade['volume'] <= remaining_volume:
                    # 完全平仓
                    remaining_volume -= oldest_trade['volume']
                    self.trade_state['open_trades'].pop(0)
                else:
                    # 部分平仓
                    oldest_trade['volume'] -= remaining_volume
                    remaining_volume = 0

        return trade_record

    def check_trade_failure(self, minute_data, trade_time, trade_type, price_level):
        """
        检查交易是否失败

        Args:
            minute_data: DataFrame，分钟级数据
            trade_time: datetime，交易时间
            trade_type: str，交易类型，'buy'或'sell'
            price_level: float，交易价格水平

        Returns:
            bool: 交易是否失败
            str: 失败原因
        """
        # 获取交易后的数据
        post_trade_data = minute_data.loc[minute_data.index > trade_time]

        if post_trade_data.empty:
            return False, "No data after trade"

        # 检查条件A：价格出现相反方向的运动
        if trade_type == 'buy':
            price_moved_opposite = post_trade_data['low'].min() < price_level
        else:
            price_moved_opposite = post_trade_data['high'].max() > price_level

        # 检查条件B：依据的交易周期背离信号消失
        if trade_type == 'buy':
            divergence_disappeared = not post_trade_data['bullish_divergence'].any()
        else:
            divergence_disappeared = not post_trade_data['bearish_divergence'].any()

        # 获取5分钟周期数据
        # 注意：这里假设minute_data是1分钟数据，实际应用中需要根据实际情况调整
        trade_minute = trade_time.minute
        next_5min_boundary = trade_time.replace(minute=(trade_minute // 5 + 1) * 5, second=0, microsecond=0)

        # 检查条件C：上一层周期的指标未出现背离
        # 这里需要等待5分钟周期完成
        if next_5min_boundary > minute_data.index[-1]:
            # 5分钟周期尚未完成
            return False, "Higher timeframe not completed yet"

        # 获取完成的5分钟周期数据
        five_min_data = minute_data.loc[minute_data.index <= next_5min_boundary].iloc[-5:]

        # 检查5分钟周期是否出现背离
        if trade_type == 'buy':
            higher_tf_divergence = five_min_data['bullish_divergence'].any()
        else:
            higher_tf_divergence = five_min_data['bearish_divergence'].any()

        # 判断交易是否失败
        if price_moved_opposite and divergence_disappeared and not higher_tf_divergence:
            return True, "Price moved opposite, divergence disappeared, and no higher timeframe divergence"

        return False, "Trade not failed"

    def handle_failed_trade(self, trade_record, current_price):
        """
        处理失败的交易

        Args:
            trade_record: dict，交易记录
            current_price: float，当前价格

        Returns:
            dict: 止损交易记录
        """
        # 执行反向交易进行止损
        reverse_type = 'sell' if trade_record['type'] == 'buy' else 'buy'

        # 执行止损交易
        stop_loss_record = self.execute_trade(
            signal_time=datetime.now(),
            signal_type=reverse_type,
            price=current_price,
            volume=trade_record['volume'],
            position_size=self.current_holdings
        )

        # 标记为止损交易
        stop_loss_record['is_stop_loss'] = True
        stop_loss_record['original_trade_time'] = trade_record['time']

        return stop_loss_record

    def get_trade_history(self):
        """
        获取交易历史记录

        Returns:
            DataFrame: 交易历史记录
        """
        return pd.DataFrame(self.trade_history)

    def get_current_holdings(self):
        """
        获取当前持仓

        Returns:
            float: 当前持仓
        """
        return self.current_holdings

    def get_current_cash(self):
        """
        获取当前现金

        Returns:
            float: 当前现金
        """
        return self.current_cash
