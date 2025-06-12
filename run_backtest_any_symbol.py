#!/usr/bin/env python3
"""
通用T0策略回测脚本 - 支持任意标的
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher
from t0t_trading_system.strategy.t0.t0_trader import ImprovedT0Trader
from t0t_trading_system.strategy.technical.indicators import TechnicalIndicators

def get_available_symbols():
    """获取可用的股票代码列表"""
    data_dir = "data/fixed_processed"
    available_symbols = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv') and file.startswith('SH') and file != 'SH000001.csv':
                symbol = file.replace('.csv', '')
                available_symbols.append(symbol)
    
    return sorted(available_symbols)

def get_stock_name(symbol):
    """根据股票代码获取股票名称"""
    stock_names = {
        'SH600036': '招商银行',
        'SH600000': '浦发银行', 
        'SH600519': '贵州茅台',
        'SH600030': '中信证券',
        'SH600887': '伊利股份',
        'SH600276': '恒瑞医药',
        'SH600585': '海螺水泥',
        'SH600104': '上汽集团',
        'SH600050': '中国联通'
    }
    return stock_names.get(symbol, symbol)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/improved_t0_backtest.log'),
            logging.StreamHandler()
        ]
    )

def load_config(symbol):
    """加载配置"""
    config = {
        'data': {
            'source': 'local',
            'local_data_dir': 'data/fixed_processed'
        },
        'backtest': {
            'start_date': '2024-04-13',  # 提前2天用于指标预热
            'end_date': '2025-05-12',
            'trading_start_date': '2024-04-15',  # 实际交易开始日期
            'initial_capital': 1000000,
            'base_position_ratio': 0.5
        },
        't0_trading': {
            'min_trade_portion': 0.1,
            'max_trade_portion': 0.3,
            'price_threshold': 0.01,
            'transaction_cost_rate': 0.0014,
            'min_trade_interval': 5,
            'max_consecutive_trades': 2,
            'min_price_change': 0.003
        },
        'symbols': [symbol]
    }
    return config

def run_t0_backtest_for_symbol(symbol):
    """运行指定标的的T0交易回测"""
    stock_name = get_stock_name(symbol)
    print("=" * 60)
    print(f"T0交易系统回测 - {stock_name} ({symbol})")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = load_config(symbol)
    
    # 检查数据文件是否存在
    data_file = f"data/fixed_processed/{symbol}.csv"
    if not os.path.exists(data_file):
        print(f"❌ 错误: 数据文件不存在 {data_file}")
        return False
    
    # 初始化数据获取器
    print("初始化数据获取器...")
    data_fetcher = MarketDataFetcher(config['data'], data_source="local")
    
    # 获取数据
    print("获取数据...")
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    
    # 获取分钟数据
    minute_data = data_fetcher.get_stock_data(
        symbol=symbol, 
        freq="min", 
        start_date=start_date, 
        end_date=end_date, 
        adjust="qfq"
    )
    
    if minute_data is None or minute_data.empty:
        print("错误: 无法获取数据")
        return False
    
    print(f"数据范围: {minute_data.index[0]} 到 {minute_data.index[-1]}")
    print(f"数据点数: {len(minute_data)}")
    
    # 初始化T0交易器
    print("初始化T0交易器...")
    t0_trader = ImprovedT0Trader(config['t0_trading'])
    
    # 设置基础仓位
    initial_capital = config['backtest']['initial_capital']
    base_position_ratio = config['backtest']['base_position_ratio']
    initial_price = minute_data['close'].iloc[0]
    base_position = (initial_capital * base_position_ratio) / initial_price
    
    t0_trader.set_base_position(base_position)
    t0_trader.current_cash = initial_capital * (1 - base_position_ratio)
    
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"基础仓位: {base_position:.3f} 股")
    print(f"初始现金: {t0_trader.current_cash:,.2f}")
    
    # 添加技术指标
    print("计算技术指标...")
    minute_data = TechnicalIndicators.calculate_macd(minute_data)
    minute_data = TechnicalIndicators.calculate_kdj(minute_data)
    
    # 运行回测
    print("开始回测...")
    all_trades = []
    daily_performance = []
    equity_curve = []

    # 按日期分组处理
    minute_data['date'] = minute_data.index.date
    dates = minute_data['date'].unique()

    # 获取实际交易开始日期
    trading_start_date = pd.to_datetime(config['backtest']['trading_start_date']).date()
    print(f"预热期: {dates[0]} 到 {trading_start_date}")
    print(f"交易期: {trading_start_date} 到 {dates[-1]}")

    for i, date in enumerate(dates):
        if i % 50 == 0:
            print(f"处理进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")

        # 获取当日数据
        day_data = minute_data[minute_data['date'] == date].copy()

        if len(day_data) < 10:  # 数据点太少，跳过
            continue

        # 判断是否在预热期
        is_warmup_period = date < trading_start_date
        
        # 重置日内状态
        t0_trader.reset_daily_state()

        # 设置当前日期（用于智能平仓决策）
        t0_trader.current_date = date

        # 检测T0交易信号（预热期也需要计算信号，用于指标计算）
        day_data_with_signals = t0_trader.detect_t0_signals(day_data)

        # 更新日内高低点
        if len(day_data_with_signals) > 0:
            t0_trader.trade_state['daily_high'] = day_data_with_signals['high'].max()
            t0_trader.trade_state['daily_low'] = day_data_with_signals['low'].min()

        # 只在非预热期执行交易
        if not is_warmup_period:
            # 执行T0交易
            for j in range(len(day_data_with_signals)):
                current_bar = day_data_with_signals.iloc[j]
                current_time = day_data_with_signals.index[j]

                # 执行T0买入（使用开盘价交易）
                if current_bar['t0_buy_signal']:
                    trade_record = t0_trader.execute_t0_trade(
                        current_time, 't0_buy', current_bar['open'], current_bar['signal_strength']
                    )
                    if trade_record:
                        all_trades.append(trade_record)

                # 执行T0卖出（使用开盘价交易）
                if current_bar['t0_sell_signal']:
                    trade_record = t0_trader.execute_t0_trade(
                        current_time, 't0_sell', current_bar['open'], current_bar['signal_strength']
                    )
                    if trade_record:
                        all_trades.append(trade_record)
        
        # 14:50强制平衡仓位（避免集合竞价阶段交易，仅在交易期执行）
        if not is_warmup_period and len(day_data_with_signals) > 0:
            # 找到14:50的数据点，如果没有则使用最后一个数据点
            force_balance_time = None
            force_balance_price = None

            for idx, row in day_data_with_signals.iterrows():
                if idx.time() >= pd.Timestamp('14:50:00').time():
                    force_balance_time = idx
                    force_balance_price = row['close']
                    break

            # 如果没有找到14:50的数据，使用最后一个数据点
            if force_balance_time is None:
                force_balance_time = day_data_with_signals.index[-1]
                force_balance_price = day_data_with_signals['close'].iloc[-1]

            balance_trade = t0_trader.force_position_balance(force_balance_price, force_balance_time, "force_balance_1450")
            if balance_trade:
                all_trades.append(balance_trade)
        
        # 记录当日表现（仅在交易期记录）
        if not is_warmup_period:
            daily_perf = t0_trader.get_daily_performance()
            daily_perf['date'] = date
            daily_perf['closing_price'] = day_data_with_signals['close'].iloc[-1] if len(day_data_with_signals) > 0 else 0
            daily_performance.append(daily_perf)

            # 计算当日权益
            current_equity = t0_trader.current_cash + t0_trader.current_holdings * daily_perf['closing_price']
            equity_curve.append({
                'date': date,
                'equity': current_equity,
                'holdings': t0_trader.current_holdings,
                'cash': t0_trader.current_cash
            })
    
    print("回测完成!")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("回测结果分析")
    print("=" * 60)
    
    # 基本统计
    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(equity_curve)
    daily_perf_df = pd.DataFrame(daily_performance)
    
    if not trades_df.empty:
        print(f"总交易次数: {len(trades_df)}")
        print(f"买入交易: {len(trades_df[trades_df['type'] == 'buy'])}")
        print(f"卖出交易: {len(trades_df[trades_df['type'] == 'sell'])}")
        print(f"T0交易: {len(trades_df[trades_df.get('is_t0_trade', False) == True])}")
        print(f"强制调整: {len(trades_df[trades_df.get('is_forced_adjustment', False) == True])}")
    
    if not equity_df.empty:
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        print(f"\n初始权益: {initial_equity:,.2f}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        
        # 计算最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].max() * 100
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        # T0交易统计
        total_t0_profit = daily_perf_df['t0_profit'].sum()
        print(f"T0交易累计收益: {total_t0_profit:.2f}")
        
        # 交易频率
        avg_daily_trades = daily_perf_df['trades_count'].mean()
        print(f"平均每日交易次数: {avg_daily_trades:.1f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists('results'):
        os.makedirs('results')

    # 添加标的信息到所有结果文件
    if not trades_df.empty:
        trades_df['symbol'] = symbol
        trades_df['stock_name'] = stock_name
        trades_file = f'results/improved_t0_trades_{timestamp}.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"\n交易记录已保存到: {trades_file}")

    if not equity_df.empty:
        equity_df['symbol'] = symbol
        equity_df['stock_name'] = stock_name
        equity_file = f'results/improved_t0_equity_{timestamp}.csv'
        equity_df.to_csv(equity_file, index=False)
        print(f"权益曲线已保存到: {equity_file}")

    if not daily_perf_df.empty:
        daily_perf_df['symbol'] = symbol
        daily_perf_df['stock_name'] = stock_name
        perf_file = f'results/improved_t0_performance_{timestamp}.csv'
        daily_perf_df.to_csv(perf_file, index=False)
        print(f"日度表现已保存到: {perf_file}")

    # 创建一个元数据文件，记录回测信息
    metadata = {
        'timestamp': timestamp,
        'symbol': symbol,
        'stock_name': stock_name,
        'backtest_period': f"{start_date} to {end_date}",
        'total_trades': len(trades_df) if not trades_df.empty else 0,
        'final_equity': final_equity if not equity_df.empty else 0,
        'total_return_pct': total_return if not equity_df.empty else 0
    }

    metadata_file = f'results/backtest_metadata_{timestamp}.json'
    import json
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"回测元数据已保存到: {metadata_file}")
    
    return True

def main():
    """主函数"""
    print("🚀 通用T0策略回测系统")
    print("="*80)
    
    # 获取可用标的
    available_symbols = get_available_symbols()
    
    if not available_symbols:
        print("❌ 未找到可用的股票数据")
        return
    
    print(f"📊 发现 {len(available_symbols)} 个可用标的:")
    for i, symbol in enumerate(available_symbols, 1):
        stock_name = get_stock_name(symbol)
        print(f"  {i}. {stock_name} ({symbol})")
    
    # 用户选择
    print(f"\n请选择要回测的标的 (1-{len(available_symbols)}):")
    try:
        choice = int(input("输入选择: ").strip())
        if 1 <= choice <= len(available_symbols):
            selected_symbol = available_symbols[choice - 1]
            stock_name = get_stock_name(selected_symbol)
            print(f"\n🎯 已选择: {stock_name} ({selected_symbol})")
            
            # 运行回测
            success = run_t0_backtest_for_symbol(selected_symbol)
            
            if success:
                print(f"\n✅ {stock_name} ({selected_symbol}) 回测完成！")
                print("💡 现在可以在Streamlit应用中查看回测结果")
            else:
                print(f"\n❌ {stock_name} ({selected_symbol}) 回测失败")
        else:
            print("❌ 无效的选择")
    except ValueError:
        print("❌ 无效的输入")
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")

if __name__ == "__main__":
    main()
