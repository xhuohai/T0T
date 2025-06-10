"""
技术指标计算模块
实现各种技术指标的计算，包括MACD、KDJ、均线等
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_ma(data, periods, price_key='close'):
        """
        计算移动平均线
        
        Args:
            data: DataFrame，包含价格数据
            periods: list，均线周期列表
            price_key: str，价格列名，默认为'close'
            
        Returns:
            DataFrame: 包含原始数据和均线数据
        """
        df = data.copy()
        
        for period in periods:
            df[f'ma_{period}'] = df[price_key].rolling(window=period).mean()
            
        return df
    
    @staticmethod
    def calculate_ema(data, periods, price_key='close'):
        """
        计算指数移动平均线
        
        Args:
            data: DataFrame，包含价格数据
            periods: list，均线周期列表
            price_key: str，价格列名，默认为'close'
            
        Returns:
            DataFrame: 包含原始数据和EMA数据
        """
        df = data.copy()
        
        for period in periods:
            df[f'ema_{period}'] = df[price_key].ewm(span=period, adjust=False).mean()
            
        return df
    
    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price_key='close'):
        """
        计算MACD指标
        
        Args:
            data: DataFrame，包含价格数据
            fast_period: int，快线周期
            slow_period: int，慢线周期
            signal_period: int，信号线周期
            price_key: str，价格列名，默认为'close'
            
        Returns:
            DataFrame: 包含原始数据和MACD数据
        """
        df = data.copy()
        
        # 计算快线和慢线的EMA
        df['ema_fast'] = df[price_key].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df[price_key].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF、DEA和MACD
        df['macd_dif'] = df['ema_fast'] - df['ema_slow']
        df['macd_dea'] = df['macd_dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd_bar'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        # 删除中间计算列
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_kdj(data, k_period=9, d_period=3, j_period=3):
        """
        计算KDJ指标
        
        Args:
            data: DataFrame，包含价格数据
            k_period: int，K线周期
            d_period: int，D线周期
            j_period: int，J线周期
            
        Returns:
            DataFrame: 包含原始数据和KDJ数据
        """
        df = data.copy()
        
        # 计算最低价和最高价的n日滚动值
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # 计算RSV
        rsv = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # 计算K、D、J值
        df['kdj_k'] = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/j_period, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        return df
    
    @staticmethod
    def calculate_atr(data, period=14):
        """
        计算ATR指标（平均真实波幅）
        
        Args:
            data: DataFrame，包含价格数据
            period: int，ATR周期
            
        Returns:
            DataFrame: 包含原始数据和ATR数据
        """
        df = data.copy()
        
        # 计算真实波幅（True Range）
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # 删除中间计算列
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    @staticmethod
    def calculate_fibonacci_levels(high_price, low_price):
        """
        计算斐波那契回调水平
        
        Args:
            high_price: float，最高价
            low_price: float，最低价
            
        Returns:
            dict: 斐波那契水平
        """
        price_range = high_price - low_price
        
        levels = {
            "0.0": low_price,
            "0.236": low_price + 0.236 * price_range,
            "0.382": low_price + 0.382 * price_range,
            "0.5": low_price + 0.5 * price_range,
            "0.618": low_price + 0.618 * price_range,
            "0.786": low_price + 0.786 * price_range,
            "1.0": high_price,
            "1.272": high_price + 0.272 * price_range,
            "1.382": high_price + 0.382 * price_range,
            "1.618": high_price + 0.618 * price_range
        }
        
        return levels
    
    @staticmethod
    def detect_divergence(data, price_key='close', indicator_key=None, window=5, threshold=0.01):
        """
        检测顶底背离（改进版）

        Args:
            data: DataFrame，包含价格和指标数据
            price_key: str，价格列名
            indicator_key: str，指标列名
            window: int，检测窗口大小
            threshold: float，背离阈值

        Returns:
            DataFrame: 包含背离信号的DataFrame
        """
        if indicator_key is None:
            raise ValueError("indicator_key must be specified")

        df = data.copy()

        # 初始化背离列
        df['bullish_divergence'] = False  # 底背离（价格创新低，指标不创新低）
        df['bearish_divergence'] = False  # 顶背离（价格创新高，指标不创新高）
        df['divergence_strength'] = 0.0  # 背离强度（使用浮点数）

        # 遍历数据检测背离
        for i in range(window * 2, len(df)):  # 需要更多历史数据来确认背离
            # 获取当前窗口的数据
            current_window = df.iloc[i-window:i+1]
            previous_window = df.iloc[i-window*2:i-window+1]

            current_price = current_window[price_key].iloc[-1]
            current_indicator = current_window[indicator_key].iloc[-1]

            # 检测底背离
            # 条件1：当前价格创新低
            current_min_price = current_window[price_key].min()
            previous_min_price = previous_window[price_key].min()

            if current_price <= current_min_price * (1 + threshold):
                # 条件2：当前价格低于前一个窗口的最低价
                if current_price < previous_min_price * (1 - threshold):
                    # 条件3：指标没有创新低，甚至可能走高
                    current_min_indicator = current_window[indicator_key].min()
                    previous_min_indicator = previous_window[indicator_key].min()

                    if current_indicator > current_min_indicator * (1 + threshold):
                        if current_indicator >= previous_min_indicator * (1 - threshold):
                            df.loc[df.index[i], 'bullish_divergence'] = True
                            # 计算背离强度
                            price_decline = (previous_min_price - current_price) / previous_min_price
                            indicator_improvement = (current_indicator - previous_min_indicator) / abs(previous_min_indicator)
                            df.loc[df.index[i], 'divergence_strength'] = price_decline + indicator_improvement

            # 检测顶背离
            # 条件1：当前价格创新高
            current_max_price = current_window[price_key].max()
            previous_max_price = previous_window[price_key].max()

            if current_price >= current_max_price * (1 - threshold):
                # 条件2：当前价格高于前一个窗口的最高价
                if current_price > previous_max_price * (1 + threshold):
                    # 条件3：指标没有创新高，甚至可能走低
                    current_max_indicator = current_window[indicator_key].max()
                    previous_max_indicator = previous_window[indicator_key].max()

                    if current_indicator < current_max_indicator * (1 - threshold):
                        if current_indicator <= previous_max_indicator * (1 + threshold):
                            df.loc[df.index[i], 'bearish_divergence'] = True
                            # 计算背离强度
                            price_rise = (current_price - previous_max_price) / previous_max_price
                            indicator_decline = (previous_max_indicator - current_indicator) / abs(previous_max_indicator)
                            df.loc[df.index[i], 'divergence_strength'] = price_rise + indicator_decline

        return df
    
    @staticmethod
    def detect_macd_divergence(data, window=5, threshold=0.01):
        """
        检测MACD顶底背离
        
        Args:
            data: DataFrame，包含价格和MACD数据
            window: int，检测窗口大小
            threshold: float，背离阈值
            
        Returns:
            DataFrame: 包含MACD背离信号的DataFrame
        """
        # 确保数据中包含MACD指标
        if 'macd_dif' not in data.columns:
            raise ValueError("Data must contain MACD indicators")
            
        return TechnicalIndicators.detect_divergence(data, 'close', 'macd_dif', window, threshold)
    
    @staticmethod
    def detect_kdj_divergence(data, window=5, threshold=0.01):
        """
        检测KDJ顶底背离
        
        Args:
            data: DataFrame，包含价格和KDJ数据
            window: int，检测窗口大小
            threshold: float，背离阈值
            
        Returns:
            DataFrame: 包含KDJ背离信号的DataFrame
        """
        # 确保数据中包含KDJ指标
        if 'kdj_j' not in data.columns:
            raise ValueError("Data must contain KDJ indicators")
            
        return TechnicalIndicators.detect_divergence(data, 'close', 'kdj_j', window, threshold)
    
    @staticmethod
    def detect_combined_divergence(data, window=5, threshold=0.01):
        """
        检测MACD和KDJ共同的顶底背离
        
        Args:
            data: DataFrame，包含价格、MACD和KDJ数据
            window: int，检测窗口大小
            threshold: float，背离阈值
            
        Returns:
            DataFrame: 包含组合背离信号的DataFrame
        """
        # 确保数据中包含MACD和KDJ指标
        required_columns = ['macd_dif', 'kdj_j']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Data must contain {col}")
        
        # 检测MACD背离
        macd_div = TechnicalIndicators.detect_macd_divergence(data, window, threshold)
        
        # 检测KDJ背离
        kdj_div = TechnicalIndicators.detect_kdj_divergence(data, window, threshold)
        
        # 组合背离信号
        df = data.copy()
        df['bullish_divergence'] = macd_div['bullish_divergence'] & kdj_div['bullish_divergence']
        df['bearish_divergence'] = macd_div['bearish_divergence'] & kdj_div['bearish_divergence']
        
        return df
