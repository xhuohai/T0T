#!/usr/bin/env python3
"""
分析回测结果脚本
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_results(results_file):
    """加载回测结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = yaml.safe_load(f)
    return results

def analyze_performance(results):
    """分析性能指标"""
    print("=" * 60)
    print("回测结果分析")
    print("=" * 60)
    
    # 获取性能指标
    metrics = results.get('performance_metrics', {})
    
    print("性能指标:")
    print("-" * 30)
    
    if 'total_return' in metrics:
        total_return = float(metrics['total_return'])
        print(f"总收益率: {total_return:.2%}")
    
    if 'annual_return' in metrics:
        annual_return = float(metrics['annual_return'])
        print(f"年化收益率: {annual_return:.2%}")
    
    if 'max_drawdown' in metrics:
        max_drawdown = float(metrics['max_drawdown'])
        print(f"最大回撤: {max_drawdown:.2%}")
    
    if 'sharpe_ratio' in metrics:
        sharpe_ratio = float(metrics['sharpe_ratio'])
        print(f"夏普比率: {sharpe_ratio:.2f}")
    
    if 'win_rate' in metrics:
        win_rate = float(metrics['win_rate'])
        print(f"胜率: {win_rate:.2%}")
    
    if 'profit_loss_ratio' in metrics:
        profit_loss_ratio = float(metrics['profit_loss_ratio'])
        print(f"盈亏比: {profit_loss_ratio:.2f}")
    
    if 'total_trades' in metrics:
        total_trades = int(metrics['total_trades'])
        print(f"总交易次数: {total_trades}")
    
    print()

def analyze_trades(results):
    """分析交易记录"""
    trades = results.get('trades')
    if trades is None:
        print("没有交易记录数据")
        return
    
    print("交易分析:")
    print("-" * 30)
    
    # 如果trades是DataFrame对象，需要特殊处理
    if hasattr(trades, 'shape'):
        print(f"交易记录数量: {len(trades)}")
        
        # 尝试获取一些基本统计
        if 'type' in trades.columns:
            buy_trades = len(trades[trades['type'] == 'buy'])
            sell_trades = len(trades[trades['type'] == 'sell'])
            print(f"买入交易: {buy_trades}")
            print(f"卖出交易: {sell_trades}")
        
        if 'value' in trades.columns:
            avg_trade_value = trades['value'].mean()
            print(f"平均交易金额: {avg_trade_value:.2f}")
    
    print()

def analyze_equity_curve(results):
    """分析权益曲线"""
    equity_curve = results.get('equity_curve')
    if equity_curve is None:
        print("没有权益曲线数据")
        return
    
    print("权益曲线分析:")
    print("-" * 30)
    
    # 如果equity_curve是DataFrame对象
    if hasattr(equity_curve, 'shape'):
        print(f"数据点数量: {len(equity_curve)}")
        
        if 'equity' in equity_curve.columns:
            initial_equity = equity_curve['equity'].iloc[0]
            final_equity = equity_curve['equity'].iloc[-1]
            print(f"初始权益: {initial_equity:.2f}")
            print(f"最终权益: {final_equity:.2f}")
            print(f"权益变化: {final_equity - initial_equity:.2f}")
    
    print()

def create_summary_report(results_file):
    """创建汇总报告"""
    results = load_results(results_file)
    
    print(f"回测结果文件: {results_file}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 分析各个部分
    analyze_performance(results)
    analyze_trades(results)
    analyze_equity_curve(results)
    
    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    
    metrics = results.get('performance_metrics', {})
    
    if 'total_return' in metrics:
        total_return = float(metrics['total_return'])
        if total_return > 0:
            print("✓ 策略产生了正收益")
        else:
            print("✗ 策略产生了负收益")
    
    if 'sharpe_ratio' in metrics:
        sharpe_ratio = float(metrics['sharpe_ratio'])
        if sharpe_ratio > 1:
            print("✓ 夏普比率良好 (>1)")
        elif sharpe_ratio > 0:
            print("△ 夏普比率一般 (0-1)")
        else:
            print("✗ 夏普比率较差 (<0)")
    
    if 'max_drawdown' in metrics:
        max_drawdown = float(metrics['max_drawdown'])
        if max_drawdown < 0.1:
            print("✓ 最大回撤控制良好 (<10%)")
        elif max_drawdown < 0.2:
            print("△ 最大回撤中等 (10-20%)")
        else:
            print("✗ 最大回撤较大 (>20%)")
    
    if 'win_rate' in metrics:
        win_rate = float(metrics['win_rate'])
        if win_rate > 0.5:
            print("✓ 胜率超过50%")
        else:
            print("△ 胜率低于50%")

def main():
    """主函数"""
    # 查找最新的结果文件
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("结果目录不存在")
        return
    
    result_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_results_') and f.endswith('.yaml')]
    
    if not result_files:
        print("没有找到回测结果文件")
        return
    
    # 使用最新的文件
    latest_file = sorted(result_files)[-1]
    results_file = os.path.join(results_dir, latest_file)
    
    print("T0T交易系统回测结果分析")
    print("=" * 60)
    
    try:
        create_summary_report(results_file)
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
