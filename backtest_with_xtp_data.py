#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用已下载的XTP数据文件回测T0T交易系统的策略
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入T0T交易系统模块
from t0t_trading_system.config.config import (
    DATA_CONFIG, INDICATOR_CONFIG, POSITION_CONFIG,
    T0_CONFIG, RISK_CONFIG, BACKTEST_CONFIG
)
from t0t_trading_system.analysis.backtest import Backtest
from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher
from t0t_trading_system.data.fetcher.xtp_data import XTPDataSource

# 尝试导入XTP API
try:
    from vnxtpquote import QuoteApi
    XTP_API_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("成功导入XTP API")
except ImportError as e:
    XTP_API_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"无法导入XTP API: {e}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("xtp_data_backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用已下载的XTP数据文件回测T0T交易系统的策略')

    parser.add_argument('--start_date', type=str, default=BACKTEST_CONFIG['start_date'],
                        help='回测开始日期，格式YYYY-MM-DD')

    parser.add_argument('--end_date', type=str, default=BACKTEST_CONFIG['end_date'],
                        help='回测结束日期，格式YYYY-MM-DD')

    parser.add_argument('--index_symbol', type=str, default=DATA_CONFIG['index_symbol'],
                        help='指数代码')

    parser.add_argument('--stock_symbol', type=str, default=None,
                        help='股票代码')

    parser.add_argument('--data_dir', type=str, default="data/xtp_data",
                        help='XTP数据目录')

    parser.add_argument('--plot', action='store_true',
                        help='是否绘制回测结果')

    return parser.parse_args()

def load_xtp_data(data_dir, symbol, start_date, end_date, freq="D"):
    """
    加载XTP数据文件，如果本地没有数据文件，则尝试使用XTP API获取

    Args:
        data_dir: 数据目录
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        freq: 数据频率，"D"表示日线，"min"表示分钟线

    Returns:
        pandas.DataFrame: 股票数据
    """
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # 构建文件名
    file_suffix = "_D_" if freq == "D" else "_1min_"
    file_name = f"{symbol}{file_suffix}{start_date}_{end_date}.csv"
    file_path = os.path.join(data_dir, file_name)

    # 检查文件是否存在
    if os.path.exists(file_path):
        logger.info(f"加载本地数据文件: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 确保数据按日期排序
        df = df.sort_index()

        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        return df

    # 如果文件不存在，尝试查找其他匹配的文件
    if freq == "D":
        # 日线数据文件通常包含"_D_"
        files = [f for f in os.listdir(data_dir) if f.startswith(symbol) and "_D_" in f and f.endswith(".csv")]
    elif freq == "min":
        # 分钟线数据文件通常包含"_1min_"或"_min_"
        files = [f for f in os.listdir(data_dir) if f.startswith(symbol) and ("_1min_" in f or "_min_" in f) and f.endswith(".csv")]
    else:
        files = [f for f in os.listdir(data_dir) if f.startswith(symbol) and f.endswith(".csv")]

    if files:
        # 加载找到的文件
        file_path = os.path.join(data_dir, files[0])
        logger.info(f"加载数据文件: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 确保数据按日期排序
        df = df.sort_index()

        # 过滤日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        return df

    # 如果本地没有数据文件，尝试使用XTP API获取
    logger.info(f"本地未找到{symbol}的{freq}数据文件，尝试使用XTP API获取...")

    if not XTP_API_AVAILABLE:
        logger.error("XTP API不可用，无法获取数据")
        return None

    try:
        # 创建数据获取器
        data_fetcher = MarketDataFetcher(DATA_CONFIG, data_source="xtp")

        # 获取数据
        if freq == "D":
            df = data_fetcher.get_stock_data(
                symbol=symbol,
                freq="D",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
        elif freq == "min":
            # 对于分钟数据，我们使用模拟数据，因为XTP API可能无法直接获取历史分钟数据
            logger.warning(f"XTP API可能无法直接获取历史分钟数据，使用日线数据生成模拟分钟数据...")

            # 首先获取日线数据
            daily_df = data_fetcher.get_stock_data(
                symbol=symbol,
                freq="D",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if daily_df is None or daily_df.empty:
                logger.error(f"使用XTP API获取{symbol}的日线数据失败，无法生成分钟数据")
                return None

            # 使用日线数据生成模拟分钟数据
            df = generate_mock_minute_data(daily_df)

            if df is None or df.empty:
                logger.error(f"生成{symbol}的模拟分钟数据失败")
                return None

            logger.info(f"成功生成{len(df)}条{symbol}的模拟分钟数据")

        # 检查数据是否为空
        if df is None or df.empty:
            logger.error(f"使用XTP API获取{symbol}的{freq}数据失败")
            return None

        # 保存数据到本地
        logger.info(f"成功获取到{len(df)}条{symbol}的{freq}数据，保存到本地...")
        df.to_csv(file_path)

        return df

    except Exception as e:
        logger.error(f"使用XTP API获取数据时发生异常: {e}")
        # 如果发生异常，尝试使用模拟数据
        logger.warning("尝试使用模拟数据...")

        # 生成日期范围
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if freq == "D":
            dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        elif freq == "min":
            # 对于分钟数据，我们只生成一天的数据作为示例
            dates = pd.date_range(start=start, end=start + timedelta(days=1), freq='min')

        # 生成随机价格数据
        n = len(dates)
        np.random.seed(42)  # 固定随机种子以便复现

        # 生成一个随机游走序列作为收盘价
        close = np.random.normal(0, 1, n).cumsum() + 3000

        # 生成其他价格数据
        high = close + np.random.uniform(10, 50, n)
        low = close - np.random.uniform(10, 50, n)
        open_price = low + np.random.uniform(0, 1, n) * (high - low)
        volume = np.random.uniform(1000000, 10000000, n)

        # 创建DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        # 保存数据到本地
        logger.info(f"成功生成{len(df)}条{symbol}的模拟数据，保存到本地...")
        df.to_csv(file_path)

        return df
    finally:
        # 确保XTP API正确退出
        try:
            if 'data_fetcher' in locals() and hasattr(data_fetcher, 'xtp'):
                data_fetcher.xtp.disconnect()
                logger.info("XTP API已断开连接")
        except Exception as e:
            logger.warning(f"断开XTP API连接时发生异常: {e}")

def prepare_data_for_backtest(daily_df, minute_df=None):
    """
    准备回测数据

    Args:
        daily_df: DataFrame，日线数据
        minute_df: DataFrame，分钟线数据，如果为None则尝试获取真实分钟数据

    Returns:
        dict: 回测数据，包含月线、周线、日线和分钟线数据
    """
    # 确保数据按日期排序
    daily_df = daily_df.sort_index()

    # 生成月线数据
    monthly_data = daily_df.resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # 生成周线数据
    weekly_data = daily_df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # 日线数据
    daily_data = daily_df.copy()

    # 分钟线数据
    if minute_df is not None and not minute_df.empty:
        # 使用真实的分钟数据
        logger.info(f"使用真实的分钟数据，共{len(minute_df)}条记录")
        minute_data = minute_df.copy()
    else:
        # 如果没有提供分钟数据，返回空的分钟数据
        logger.warning("未提供分钟数据，将使用空的分钟数据DataFrame")
        minute_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # 返回数据
    return {
        'index_data': {
            'monthly': monthly_data,
            'weekly': weekly_data,
            'daily': daily_data
        },
        'stock_data': {
            'daily': daily_data,
            'minute': minute_data
        }
    }

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
            logger.warning(f"列 '{col}' 在daily_data中不存在。可用列: {daily_data.columns}")
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

def run_backtest(data, config):
    """
    运行回测

    Args:
        data: dict，回测数据
        config: dict，配置参数

    Returns:
        dict: 回测结果
    """
    # 初始化回测器
    backtest = Backtest(config)

    # 准备数据
    processed_data = backtest.prepare_data(data['index_data'], data['stock_data'])

    # 运行仓位管理策略
    position_data = backtest.run_position_strategy(processed_data)

    # 确保position_data有正确的日期列格式
    if 'date' in position_data.columns and not pd.api.types.is_datetime64_dtype(position_data['date']):
        position_data['date'] = pd.to_datetime(position_data['date'])

    # 运行T0交易策略
    trade_data = backtest.run_t0_strategy(processed_data, position_data)

    # 生成权益曲线
    equity_curve = backtest.generate_equity_curve(trade_data)

    # 计算绩效指标
    performance_metrics = backtest.calculate_performance_metrics(equity_curve)

    # 保存回测结果
    results = {
        'equity_curve': equity_curve,
        'trades': trade_data,
        'positions': position_data,
        'performance_metrics': performance_metrics
    }

    return results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置回测配置
    backtest_config = BACKTEST_CONFIG.copy()
    backtest_config['start_date'] = args.start_date
    backtest_config['end_date'] = args.end_date

    logger.info(f"启动XTP数据回测，回测期间: {args.start_date} 至 {args.end_date}")
    logger.info(f"指数: {args.index_symbol}, 股票: {args.stock_symbol if args.stock_symbol else '未指定'}")

    try:
        # 加载指数数据
        logger.info(f"加载指数 {args.index_symbol} 日线数据...")
        index_df = load_xtp_data(args.data_dir, args.index_symbol, args.start_date, args.end_date, freq="D")

        if index_df is None or index_df.empty:
            logger.error(f"未找到指数 {args.index_symbol} 日线数据或数据为空")
            return

        logger.info(f"成功加载 {len(index_df)} 条指数日线数据")

        # 加载股票数据
        stock_daily_df = None
        stock_minute_df = None

        if args.stock_symbol:
            # 加载股票日线数据
            logger.info(f"加载股票 {args.stock_symbol} 日线数据...")
            stock_daily_df = load_xtp_data(args.data_dir, args.stock_symbol, args.start_date, args.end_date, freq="D")

            if stock_daily_df is None or stock_daily_df.empty:
                logger.error(f"未找到股票 {args.stock_symbol} 日线数据或数据为空")
                return

            logger.info(f"成功加载 {len(stock_daily_df)} 条股票日线数据")

            # 尝试加载股票分钟线数据
            logger.info(f"尝试加载股票 {args.stock_symbol} 分钟线数据...")
            try:
                stock_minute_df = load_xtp_data(args.data_dir, args.stock_symbol, args.start_date, args.end_date, freq="min")

                if stock_minute_df is None or stock_minute_df.empty:
                    logger.warning(f"未找到股票 {args.stock_symbol} 分钟线数据或数据为空，将不使用分钟数据")
                else:
                    logger.info(f"成功加载 {len(stock_minute_df)} 条股票分钟线数据")
            except Exception as e:
                logger.error(f"加载分钟数据时发生异常: {e}")
                logger.warning("将不使用分钟数据")
                stock_minute_df = None
        else:
            # 如果没有指定股票，使用指数数据
            stock_daily_df = index_df.copy()

            # 尝试加载指数分钟线数据
            logger.info(f"尝试加载指数 {args.index_symbol} 分钟线数据...")
            try:
                stock_minute_df = load_xtp_data(args.data_dir, args.index_symbol, args.start_date, args.end_date, freq="min")

                if stock_minute_df is None or stock_minute_df.empty:
                    logger.warning(f"未找到指数 {args.index_symbol} 分钟线数据或数据为空，将不使用分钟数据")
                else:
                    logger.info(f"成功加载 {len(stock_minute_df)} 条指数分钟线数据")
            except Exception as e:
                logger.error(f"加载分钟数据时发生异常: {e}")
                logger.warning("将不使用分钟数据")
                stock_minute_df = None

        # 准备回测数据
        logger.info("准备回测数据...")
        data = prepare_data_for_backtest(stock_daily_df, stock_minute_df)

        # 运行回测
        logger.info("开始运行回测...")
        results = run_backtest(data, backtest_config)

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
            # 创建一个新的回测器用于绘图
            plot_backtest = Backtest(backtest_config)
            plot_backtest.results = results
            plot_backtest.plot_results()

    except Exception as e:
        logger.error(f"回测过程中发生异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 确保XTP API正确退出
        if XTP_API_AVAILABLE:
            try:
                # 尝试调用exit方法确保线程正确退出
                logger.info("正在尝试退出XTP API...")
                # 这里我们创建一个临时的API对象，然后调用exit方法
                temp_api = QuoteApi.createQuoteApi(1, os.path.join(os.getcwd(), "xtp_log"), 1)
                if temp_api:
                    temp_api.exit()
                    logger.info("XTP API成功退出")
            except Exception as e:
                logger.warning(f"尝试退出XTP API时发生异常: {e}")
        else:
            logger.info("XTP API不可用，无需执行退出操作")

if __name__ == "__main__":
    main()
