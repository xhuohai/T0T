"""
技术指标计算模块的单元测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from t0t_trading_system.strategy.technical.indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标计算类的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

        # 生成模拟价格数据
        np.random.seed(42)  # 固定随机种子以便测试结果可重现

        # 生成一个随机游走序列作为收盘价
        close = np.random.normal(0, 1, 100).cumsum() + 100

        # 生成其他价格数据
        high = close + np.random.uniform(1, 5, 100)
        low = close - np.random.uniform(1, 5, 100)
        open_price = low + np.random.uniform(0, 1, 100) * (high - low)
        volume = np.random.uniform(1000000, 10000000, 100)

        # 创建DataFrame
        self.test_data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        # 初始化技术指标计算器
        self.indicators = TechnicalIndicators()

    def test_calculate_ma(self):
        """测试移动平均线计算"""
        # 计算移动平均线
        periods = [5, 10, 20]
        result = self.indicators.calculate_ma(self.test_data, periods)

        # 检查结果是否包含所有均线列
        for period in periods:
            self.assertIn(f'ma_{period}', result.columns)

        # 检查计算结果是否正确
        for period in periods:
            # 手动计算均线
            expected_ma = self.test_data['close'].rolling(window=period).mean()

            # 比较计算结果（忽略名称）
            pd.testing.assert_series_equal(result[f'ma_{period}'], expected_ma, check_names=False)

    def test_calculate_ema(self):
        """测试指数移动平均线计算"""
        # 计算指数移动平均线
        periods = [5, 10, 20]
        result = self.indicators.calculate_ema(self.test_data, periods)

        # 检查结果是否包含所有EMA列
        for period in periods:
            self.assertIn(f'ema_{period}', result.columns)

        # 检查计算结果是否正确
        for period in periods:
            # 手动计算EMA
            expected_ema = self.test_data['close'].ewm(span=period, adjust=False).mean()

            # 比较计算结果（忽略名称）
            pd.testing.assert_series_equal(result[f'ema_{period}'], expected_ema, check_names=False)

    def test_calculate_macd(self):
        """测试MACD指标计算"""
        # 计算MACD
        result = self.indicators.calculate_macd(self.test_data)

        # 检查结果是否包含MACD相关列
        self.assertIn('macd_dif', result.columns)
        self.assertIn('macd_dea', result.columns)
        self.assertIn('macd_bar', result.columns)

        # 手动计算MACD
        ema_fast = self.test_data['close'].ewm(span=12, adjust=False).mean()
        ema_slow = self.test_data['close'].ewm(span=26, adjust=False).mean()
        expected_dif = ema_fast - ema_slow
        expected_dea = expected_dif.ewm(span=9, adjust=False).mean()
        expected_bar = 2 * (expected_dif - expected_dea)

        # 比较计算结果（忽略名称）
        pd.testing.assert_series_equal(result['macd_dif'], expected_dif, check_names=False)
        pd.testing.assert_series_equal(result['macd_dea'], expected_dea, check_names=False)
        pd.testing.assert_series_equal(result['macd_bar'], expected_bar, check_names=False)

    def test_calculate_kdj(self):
        """测试KDJ指标计算"""
        # 计算KDJ
        result = self.indicators.calculate_kdj(self.test_data)

        # 检查结果是否包含KDJ相关列
        self.assertIn('kdj_k', result.columns)
        self.assertIn('kdj_d', result.columns)
        self.assertIn('kdj_j', result.columns)

        # 由于KDJ计算较复杂，这里只检查值是否有效
        self.assertTrue(not result['kdj_k'].isna().all())
        self.assertTrue(not result['kdj_d'].isna().all())
        self.assertTrue(not result['kdj_j'].isna().all())

    def test_calculate_atr(self):
        """测试ATR指标计算"""
        # 计算ATR
        result = self.indicators.calculate_atr(self.test_data)

        # 检查结果是否包含ATR列
        self.assertIn('atr', result.columns)

        # 检查ATR值是否为正
        self.assertTrue((result['atr'].dropna() > 0).all())

    def test_calculate_fibonacci_levels(self):
        """测试斐波那契回调水平计算"""
        # 计算斐波那契水平
        high_price = 100
        low_price = 80
        levels = self.indicators.calculate_fibonacci_levels(high_price, low_price)

        # 检查关键水平
        self.assertAlmostEqual(levels["0.0"], 80)
        self.assertAlmostEqual(levels["0.236"], 80 + 0.236 * 20)
        self.assertAlmostEqual(levels["0.382"], 80 + 0.382 * 20)
        self.assertAlmostEqual(levels["0.5"], 80 + 0.5 * 20)
        self.assertAlmostEqual(levels["0.618"], 80 + 0.618 * 20)
        self.assertAlmostEqual(levels["1.0"], 100)
        self.assertAlmostEqual(levels["1.618"], 100 + 0.618 * 20)

    def test_detect_divergence(self):
        """测试背离检测"""
        # 创建一个包含价格和指标的测试数据
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')

        # 创建价格序列，形成两个低点，第二个低点比第一个低
        price = np.array([100] * 10 + list(range(100, 90, -1)) + [90] * 5 +
                         list(range(90, 100)) + [100] * 5 +
                         list(range(100, 85, -1)))

        # 创建指标序列，形成两个低点，第二个低点比第一个高（底背离）
        indicator = np.array([50] * 10 + list(range(50, 30, -2)) + [30] * 5 +
                            list(range(30, 50, 2)) + [50] * 5 +
                            list(range(50, 40, -1)))

        # 确保长度匹配
        price = price[:50]
        indicator = indicator[:50]

        # 创建DataFrame
        test_data = pd.DataFrame({
            'close': price,
            'indicator': indicator
        }, index=dates)

        # 检测背离
        # 由于背离检测逻辑较复杂，这里只检查函数是否正常运行
        result = self.indicators.detect_divergence(test_data, 'close', 'indicator')

        # 检查结果是否包含背离列
        self.assertIn('bullish_divergence', result.columns)
        self.assertIn('bearish_divergence', result.columns)

    def test_detect_macd_divergence(self):
        """测试MACD背离检测"""
        # 先计算MACD
        data_with_macd = self.indicators.calculate_macd(self.test_data)

        # 检测MACD背离
        result = self.indicators.detect_macd_divergence(data_with_macd)

        # 检查结果是否包含背离列
        self.assertIn('bullish_divergence', result.columns)
        self.assertIn('bearish_divergence', result.columns)

    def test_detect_kdj_divergence(self):
        """测试KDJ背离检测"""
        # 先计算KDJ
        data_with_kdj = self.indicators.calculate_kdj(self.test_data)

        # 检测KDJ背离
        result = self.indicators.detect_kdj_divergence(data_with_kdj)

        # 检查结果是否包含背离列
        self.assertIn('bullish_divergence', result.columns)
        self.assertIn('bearish_divergence', result.columns)

    def test_detect_combined_divergence(self):
        """测试组合背离检测"""
        # 先计算MACD和KDJ
        data_with_indicators = self.indicators.calculate_macd(self.test_data)
        data_with_indicators = self.indicators.calculate_kdj(data_with_indicators)

        # 检测组合背离
        result = self.indicators.detect_combined_divergence(data_with_indicators)

        # 检查结果是否包含背离列
        self.assertIn('bullish_divergence', result.columns)
        self.assertIn('bearish_divergence', result.columns)


if __name__ == '__main__':
    unittest.main()
