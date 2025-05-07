"""
主程序入口
用于运行整个交易系统
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t0t_trading_system.config.config import (
    DATA_CONFIG, INDICATOR_CONFIG, POSITION_CONFIG,
    T0_CONFIG, RISK_CONFIG, BACKTEST_CONFIG
)
from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher
from t0t_trading_system.strategy.technical.indicators import TechnicalIndicators
from t0t_trading_system.strategy.position.position_manager import PositionManager
from t0t_trading_system.strategy.t0.t0_trader import T0Trader
from t0t_trading_system.risk_management.risk_manager import RiskManager
from t0t_trading_system.analysis.backtest import Backtest


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("t0t_trading_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_argparse():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='T0T Trading System')

    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['backtest', 'live'],
                        help='运行模式: backtest或live')

    parser.add_argument('--start_date', type=str, default=BACKTEST_CONFIG['start_date'],
                        help='回测开始日期，格式YYYY-MM-DD')

    parser.add_argument('--end_date', type=str, default=BACKTEST_CONFIG['end_date'],
                        help='回测结束日期，格式YYYY-MM-DD')

    parser.add_argument('--index_symbol', type=str, default=DATA_CONFIG['index_symbol'],
                        help='指数代码')

    parser.add_argument('--stock_symbol', type=str, default=None,
                        help='股票代码')

    parser.add_argument('--data_source', type=str, default=DATA_CONFIG['data_source'],
                        choices=['tushare', 'akshare', 'baostock', 'mock'],
                        help='数据源')

    parser.add_argument('--token', type=str, default=None,
                        help='数据源API token')

    parser.add_argument('--plot', action='store_true',
                        help='是否绘制回测结果')

    return parser.parse_args()


def fetch_data(args):
    """获取数据"""
    logger.info("开始获取数据...")

    # 初始化数据获取器
    data_fetcher = MarketDataFetcher(DATA_CONFIG, args.data_source, args.token)

    # 获取指数数据
    index_data = {}

    # 获取月线数据
    logger.info("获取月线数据...")
    index_data['monthly'] = data_fetcher.get_index_data(
        symbol=args.index_symbol,
        freq="M",
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 获取周线数据
    logger.info("获取周线数据...")
    index_data['weekly'] = data_fetcher.get_index_data(
        symbol=args.index_symbol,
        freq="W",
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 获取日线数据
    logger.info("获取日线数据...")
    index_data['daily'] = data_fetcher.get_index_data(
        symbol=args.index_symbol,
        freq="D",
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 获取个股数据
    stock_data = {}

    # 如果指定了股票代码，获取个股数据
    if args.stock_symbol:
        # 获取日线数据
        logger.info(f"获取股票 {args.stock_symbol} 日线数据...")
        stock_data['daily'] = data_fetcher.get_stock_data(
            symbol=args.stock_symbol,
            freq="D",
            start_date=args.start_date,
            end_date=args.end_date,
            adjust="qfq"
        )

        # 获取分钟数据
        logger.info("获取分钟数据...")
        try:
            if args.data_source == "akshare":
                # 尝试使用akshare获取实际的分钟数据
                logger.info("尝试使用akshare获取实际分钟数据...")
                minute_data = data_fetcher.get_stock_data(
                    symbol=args.stock_symbol,
                    freq="min",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjust="qfq"
                )

                # 检查获取的分钟数据是否为空
                if minute_data is not None and not minute_data.empty and len(minute_data) > 10:
                    logger.info(f"成功获取到 {len(minute_data)} 条实际分钟数据")
                    stock_data['minute'] = minute_data
                else:
                    # 如果获取的分钟数据为空或数量太少，使用模拟数据
                    logger.info("获取的实际分钟数据为空或数量太少，使用模拟分钟数据...")
                    stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])
            else:
                # 其他数据源使用模拟分钟数据
                logger.info("使用模拟分钟数据...")
                stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])
        except Exception as e:
            # 如果获取分钟数据时发生异常，使用模拟数据
            logger.error(f"获取分钟数据时发生异常: {e}")
            logger.info("使用模拟分钟数据...")
            stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])
    else:
        # 如果没有指定股票代码，使用指数数据作为股票数据
        logger.info("未指定股票代码，使用指数数据作为股票数据...")
        stock_data['daily'] = index_data['daily'].copy()

        # 获取分钟数据
        logger.info("获取分钟数据...")
        try:
            if args.data_source == "akshare":
                # 尝试使用akshare获取实际的分钟数据
                logger.info("尝试使用akshare获取实际分钟数据...")
                minute_data = data_fetcher.get_stock_data(
                    symbol=args.index_symbol,
                    freq="min",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjust="qfq"
                )

                # 检查获取的分钟数据是否为空
                if minute_data is not None and not minute_data.empty and len(minute_data) > 10:
                    logger.info(f"成功获取到 {len(minute_data)} 条实际分钟数据")
                    stock_data['minute'] = minute_data
                else:
                    # 如果获取的分钟数据为空或数量太少，使用模拟数据
                    logger.info("获取的实际分钟数据为空或数量太少，使用模拟分钟数据...")
                    stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])
            else:
                # 其他数据源使用模拟分钟数据
                logger.info("使用模拟分钟数据...")
                stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])
        except Exception as e:
            # 如果获取分钟数据时发生异常，使用模拟数据
            logger.error(f"获取分钟数据时发生异常: {e}")
            logger.info("使用模拟分钟数据...")
            stock_data['minute'] = generate_mock_minute_data(stock_data['daily'])

    logger.info("数据获取完成")

    return index_data, stock_data


def generate_mock_minute_data(daily_data):
    """
    生成模拟分钟数据

    Args:
        daily_data: DataFrame，日线数据

    Returns:
        DataFrame: 模拟的分钟数据
    """
    minute_data_list = []

    # 交易时间段
    morning_start = pd.Timestamp('09:30:00').time()
    morning_end = pd.Timestamp('11:30:00').time()
    afternoon_start = pd.Timestamp('13:00:00').time()
    afternoon_end = pd.Timestamp('15:00:00').time()

    # 检查daily_data是否为空
    if daily_data.empty:
        # 返回空的分钟数据框架
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # 检查daily_data是否包含所需的列
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in daily_data.columns:
            print(f"Warning: Column '{col}' not found in daily_data. Available columns: {daily_data.columns}")
            # 如果缺少必要的列，使用模拟数据
            if col == 'volume':
                daily_data[col] = 1000000  # 默认成交量
            else:
                # 如果有close列，使用close填充其他价格列
                if 'close' in daily_data.columns:
                    daily_data[col] = daily_data['close']
                # 否则使用默认值
                else:
                    daily_data[col] = 100  # 默认价格

    # 为每个交易日生成分钟数据
    for date, day_data in daily_data.iterrows():
        # 确保日期是datetime类型
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        # 当日开盘价、最高价、最低价、收盘价
        open_price = day_data['open']
        high_price = day_data['high']
        low_price = day_data['low']
        close_price = day_data['close']

        # 检查价格是否有效
        if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
            continue

        # 确保价格范围合理
        if high_price <= low_price:
            high_price = low_price * 1.01

        # 生成交易时间列表
        trading_minutes = []

        # 上午交易时段
        current_time = pd.Timestamp.combine(date.date(), morning_start)
        end_time = pd.Timestamp.combine(date.date(), morning_end)

        while current_time <= end_time:
            trading_minutes.append(current_time)
            current_time += timedelta(minutes=1)

        # 下午交易时段
        current_time = pd.Timestamp.combine(date.date(), afternoon_start)
        end_time = pd.Timestamp.combine(date.date(), afternoon_end)

        while current_time <= end_time:
            trading_minutes.append(current_time)
            current_time += timedelta(minutes=1)

        # 生成价格序列
        # 使用随机游走模型生成分钟价格
        np.random.seed(int(date.timestamp()))  # 使用日期作为随机种子，确保可重复性

        # 价格波动范围
        price_range = high_price - low_price

        # 生成随机价格序列
        num_minutes = len(trading_minutes)
        random_walk = np.random.normal(0, 1, num_minutes).cumsum() * price_range / (num_minutes ** 0.5)

        # 调整价格序列，使其符合日内高低点
        price_series = open_price + random_walk

        # 调整使得最高价和最低价符合日线数据
        actual_high = max(price_series)
        actual_low = min(price_series)

        # 线性调整
        if actual_high > actual_low:  # 防止除以零
            price_series = (price_series - actual_low) / (actual_high - actual_low) * (high_price - low_price) + low_price
        else:
            # 如果所有价格相同，使用固定价格
            price_series = np.full_like(price_series, open_price)

        # 确保收盘价正确
        price_series[-1] = close_price

        # 生成分钟K线数据
        for i, minute in enumerate(trading_minutes):
            # 如果是第一分钟，使用开盘价
            if i == 0:
                minute_open = open_price
            else:
                minute_open = price_series[i-1]

            minute_close = price_series[i]

            # 分钟内的高低价
            minute_high = max(minute_open, minute_close) + np.random.uniform(0, 0.01) * price_range
            minute_low = min(minute_open, minute_close) - np.random.uniform(0, 0.01) * price_range

            # 确保不超过日内最高最低价
            minute_high = min(minute_high, high_price)
            minute_low = max(minute_low, low_price)

            # 分钟成交量，按照日内分布模拟
            if i < 30 or i > num_minutes - 30:
                # 开盘和收盘附近成交量较大
                volume_factor = np.random.uniform(1.5, 2.5)
            else:
                # 中间时段成交量较小
                volume_factor = np.random.uniform(0.5, 1.5)

            minute_volume = day_data['volume'] * volume_factor / num_minutes

            # 添加到分钟数据列表
            minute_data_list.append({
                'datetime': minute,
                'open': minute_open,
                'high': minute_high,
                'low': minute_low,
                'close': minute_close,
                'volume': minute_volume
            })

    # 转换为DataFrame
    if minute_data_list:
        minute_data = pd.DataFrame(minute_data_list)
        minute_data = minute_data.set_index('datetime')
        return minute_data
    else:
        # 如果没有生成任何分钟数据，返回空的DataFrame
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])


def run_backtest(args, index_data, stock_data):
    """运行回测"""
    logger.info("开始回测...")

    # 更新回测配置
    backtest_config = BACKTEST_CONFIG.copy()
    backtest_config['start_date'] = args.start_date
    backtest_config['end_date'] = args.end_date

    # 初始化回测器
    backtest = Backtest(backtest_config)

    # 运行回测
    results = backtest.run(index_data, stock_data)

    # 打印回测结果
    logger.info("回测完成")
    logger.info(f"总收益率: {results['performance_metrics']['total_return']:.4f}")
    logger.info(f"年化收益率: {results['performance_metrics']['annual_return']:.4f}")
    logger.info(f"最大回撤: {results['performance_metrics']['max_drawdown']:.4f}")
    logger.info(f"夏普比率: {results['performance_metrics']['sharpe_ratio']:.4f}")
    logger.info(f"胜率: {results['performance_metrics']['win_rate']:.4f}")
    logger.info(f"盈亏比: {results['performance_metrics']['profit_loss_ratio']:.4f}")
    logger.info(f"总交易次数: {results['performance_metrics']['total_trades']}")

    # 绘制回测结果
    if args.plot:
        logger.info("绘制回测结果...")
        backtest.plot_results()

    return results


def main():
    """主函数"""
    # 解析命令行参数
    args = setup_argparse()

    logger.info(f"启动T0T交易系统，运行模式: {args.mode}")

    # 获取数据
    index_data, stock_data = fetch_data(args)

    # 根据运行模式执行相应操作
    if args.mode == 'backtest':
        # 运行回测
        results = run_backtest(args, index_data, stock_data)
    elif args.mode == 'live':
        # 实时交易模式（暂未实现）
        logger.info("实时交易模式暂未实现")

    logger.info("程序执行完毕")


if __name__ == "__main__":
    main()
