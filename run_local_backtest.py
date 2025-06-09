#!/usr/bin/env python3
"""
本地数据回测脚本
使用处理好的本地CSV数据进行回测
"""

import os
import sys
import yaml
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher
from t0t_trading_system.analysis.backtest import Backtest

def setup_logging(config):
    """设置日志"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    log_file = log_config.get('file', 'logs/backtest.log')
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    
    if log_config.get('console', True):
        root_logger.addHandler(console_handler)

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def fetch_multi_stock_data(data_fetcher, symbols, start_date, end_date):
    """
    获取多只股票的数据

    Args:
        data_fetcher: 数据获取器
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        tuple: (index_data, stock_data)
    """
    print(f"获取 {len(symbols)} 只股票的数据...")

    # 使用第一只股票作为指数数据（通常是上证指数）
    index_symbol = symbols[0] if symbols else "SH000001"

    # 获取指数数据
    index_data = {}
    print(f"获取指数 {index_symbol} 数据...")

    # 获取月线数据
    index_data['monthly'] = data_fetcher.get_index_data(
        symbol=index_symbol, freq="M", start_date=start_date, end_date=end_date
    )

    # 获取周线数据
    index_data['weekly'] = data_fetcher.get_index_data(
        symbol=index_symbol, freq="W", start_date=start_date, end_date=end_date
    )

    # 获取日线数据
    index_data['daily'] = data_fetcher.get_index_data(
        symbol=index_symbol, freq="D", start_date=start_date, end_date=end_date
    )

    # 获取股票数据（使用第一只股票作为主要交易标的）
    stock_data = {}
    main_symbol = symbols[0] if symbols else index_symbol

    print(f"获取股票 {main_symbol} 数据...")

    # 获取日线数据
    stock_data['daily'] = data_fetcher.get_stock_data(
        symbol=main_symbol, freq="D", start_date=start_date, end_date=end_date, adjust="qfq"
    )

    # 获取分钟数据
    stock_data['minute'] = data_fetcher.get_stock_data(
        symbol=main_symbol, freq="min", start_date=start_date, end_date=end_date, adjust="qfq"
    )

    print("数据获取完成")
    return index_data, stock_data

def validate_data_availability(config):
    """验证数据可用性"""
    data_dir = config['data']['local_data_dir']
    symbols = config['stock_pool']['symbols']
    
    print("验证数据可用性...")
    available_symbols = []
    missing_symbols = []
    
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            available_symbols.append(symbol)
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"✓ {symbol}: {file_size:,} bytes")
        else:
            missing_symbols.append(symbol)
            print(f"✗ {symbol}: 文件不存在")
    
    print(f"\n数据统计:")
    print(f"可用股票: {len(available_symbols)}")
    print(f"缺失股票: {len(missing_symbols)}")
    
    if missing_symbols:
        print(f"缺失的股票: {missing_symbols}")
        
    return available_symbols, missing_symbols

def run_backtest(config_file):
    """运行回测"""
    print("=" * 60)
    print("T0T交易系统 - 本地数据回测")
    print("=" * 60)
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(config_file)
    
    # 设置日志
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("开始本地数据回测")
    logger.info(f"配置文件: {config_file}")
    
    # 验证数据可用性
    available_symbols, missing_symbols = validate_data_availability(config)
    
    if not available_symbols:
        print("错误: 没有可用的数据文件")
        return
    
    # 更新配置中的股票列表，只使用可用的股票
    config['stock_pool']['symbols'] = available_symbols
    
    # 创建输出目录
    output_dir = config.get('output', {}).get('output_dir', 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化数据获取器
        print("\n初始化数据获取器...")
        data_fetcher = MarketDataFetcher(config['data'], data_source="local")

        # 获取数据
        print("\n获取数据...")
        index_data, stock_data = fetch_multi_stock_data(
            data_fetcher,
            available_symbols,
            config['backtest']['start_date'],
            config['backtest']['end_date']
        )

        # 初始化回测器
        print("\n初始化回测器...")
        backtest = Backtest(config['backtest'])

        # 运行回测
        print("\n开始回测...")
        start_time = datetime.now()

        results = backtest.run(index_data, stock_data)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n回测完成! 耗时: {duration}")
        
        # 显示回测结果
        if results:
            print("\n" + "=" * 60)
            print("回测结果摘要")
            print("=" * 60)

            # 获取性能指标
            performance_metrics = results.get('performance_metrics', {})

            # 基本统计
            if 'total_return' in performance_metrics:
                print(f"总收益率: {performance_metrics['total_return']:.2%}")
            if 'annual_return' in performance_metrics:
                print(f"年化收益率: {performance_metrics['annual_return']:.2%}")
            if 'max_drawdown' in performance_metrics:
                print(f"最大回撤: {performance_metrics['max_drawdown']:.2%}")
            if 'sharpe_ratio' in performance_metrics:
                print(f"夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
            if 'win_rate' in performance_metrics:
                print(f"胜率: {performance_metrics['win_rate']:.2%}")
            if 'profit_loss_ratio' in performance_metrics:
                print(f"盈亏比: {performance_metrics['profit_loss_ratio']:.2f}")
            if 'total_trades' in performance_metrics:
                print(f"总交易次数: {performance_metrics['total_trades']}")

            # 保存简化的结果
            simple_results = {
                'performance_metrics': performance_metrics,
                'backtest_config': {
                    'start_date': config['backtest']['start_date'],
                    'end_date': config['backtest']['end_date'],
                    'initial_capital': config['backtest']['initial_capital'],
                    'symbols': available_symbols
                },
                'timestamp': datetime.now().isoformat()
            }

            results_file = os.path.join(output_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(simple_results, f, default_flow_style=False, allow_unicode=True)
            print(f"\n回测摘要已保存到: {results_file}")

            # 保存权益曲线数据
            if 'equity_curve' in results and results['equity_curve'] is not None:
                equity_file = os.path.join(output_dir, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                results['equity_curve'].to_csv(equity_file)
                print(f"权益曲线数据已保存到: {equity_file}")

            # 保存交易记录
            if 'trades' in results and results['trades'] is not None:
                trades_file = os.path.join(output_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                results['trades'].to_csv(trades_file)
                print(f"交易记录已保存到: {trades_file}")
        
        logger.info("回测完成")
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    config_file = "config/local_backtest_config.yaml"
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在 {config_file}")
        return
    
    run_backtest(config_file)

if __name__ == "__main__":
    main()
