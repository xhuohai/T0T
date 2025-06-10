#!/usr/bin/env python3
"""
改进的T0交易回测脚本
基于真实T0交易原理：维持日内仓位不变，通过低买高卖获取价差
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

def load_config():
    """加载配置"""
    config = {
        'data': {
            'source': 'local',
            'local_data_dir': 'data/cleaned'
        },
        'backtest': {
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 1000000,
            'base_position_ratio': 0.5  # 基础仓位比例50%
        },
        't0_trading': {
            'min_trade_portion': 0.1,    # 最小交易比例10%
            'max_trade_portion': 0.3,    # 最大交易比例30%
            'price_threshold': 0.01      # 价格变动阈值1%
        },
        'symbols': ['SH600036']  # 使用招商银行进行测试
    }
    return config

def run_improved_t0_backtest():
    """运行改进的T0交易回测"""
    print("=" * 60)
    print("改进的T0交易系统回测")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = load_config()
    
    # 初始化数据获取器
    print("初始化数据获取器...")
    data_fetcher = MarketDataFetcher(config['data'], data_source="local")
    
    # 获取数据
    print("获取数据...")
    symbol = config['symbols'][0]
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
        return
    
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
    
    for i, date in enumerate(dates):
        if i % 50 == 0:
            print(f"处理进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")
        
        # 获取当日数据
        day_data = minute_data[minute_data['date'] == date].copy()
        
        if len(day_data) < 10:  # 数据点太少，跳过
            continue
        
        # 重置日内状态
        t0_trader.reset_daily_state()
        
        # 检测T0交易信号
        day_data_with_signals = t0_trader.detect_t0_signals(day_data)
        
        # 执行T0交易
        for j in range(len(day_data_with_signals)):
            current_bar = day_data_with_signals.iloc[j]
            current_time = day_data_with_signals.index[j]
            
            # 执行T0买入
            if current_bar['t0_buy_signal']:
                trade_record = t0_trader.execute_t0_trade(
                    current_time, 't0_buy', current_bar['close'], current_bar['signal_strength']
                )
                if trade_record:
                    all_trades.append(trade_record)
            
            # 执行T0卖出
            if current_bar['t0_sell_signal']:
                trade_record = t0_trader.execute_t0_trade(
                    current_time, 't0_sell', current_bar['close'], current_bar['signal_strength']
                )
                if trade_record:
                    all_trades.append(trade_record)
        
        # 收盘前强制平衡仓位
        if len(day_data_with_signals) > 0:
            last_price = day_data_with_signals['close'].iloc[-1]
            balance_trade = t0_trader.force_position_balance(last_price, "end_of_day")
            if balance_trade:
                all_trades.append(balance_trade)
        
        # 记录当日表现
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
    
    if not trades_df.empty:
        trades_file = f'results/improved_t0_trades_{timestamp}.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"\n交易记录已保存到: {trades_file}")
    
    if not equity_df.empty:
        equity_file = f'results/improved_t0_equity_{timestamp}.csv'
        equity_df.to_csv(equity_file, index=False)
        print(f"权益曲线已保存到: {equity_file}")
    
    if not daily_perf_df.empty:
        perf_file = f'results/improved_t0_performance_{timestamp}.csv'
        daily_perf_df.to_csv(perf_file, index=False)
        print(f"日度表现已保存到: {perf_file}")

def main():
    """主函数"""
    try:
        run_improved_t0_backtest()
    except Exception as e:
        print(f"回测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
