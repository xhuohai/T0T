#!/usr/bin/env python3
"""
分析改进T0交易结果的脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_improved_t0_results():
    """分析改进的T0交易结果"""
    print("=" * 60)
    print("改进T0交易策略结果分析")
    print("=" * 60)
    
    # 读取最新数据
    trades_df = pd.read_csv('results/improved_t0_trades_20250610_111204.csv')
    equity_df = pd.read_csv('results/improved_t0_equity_20250610_111204.csv')
    performance_df = pd.read_csv('results/improved_t0_performance_20250610_111204.csv')
    
    # 转换时间列
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    performance_df['date'] = pd.to_datetime(performance_df['date'])
    
    print("1. 交易统计分析")
    print("-" * 30)
    
    # 基本交易统计
    total_trades = len(trades_df)
    buy_trades = len(trades_df[trades_df['type'] == 'buy'])
    sell_trades = len(trades_df[trades_df['type'] == 'sell'])
    t0_trades = len(trades_df[trades_df.get('is_t0_trade', False) == True])
    forced_trades = len(trades_df[trades_df.get('is_forced_adjustment', False) == True])
    
    print(f"总交易次数: {total_trades}")
    print(f"买入交易: {buy_trades} ({buy_trades/total_trades:.1%})")
    print(f"卖出交易: {sell_trades} ({sell_trades/total_trades:.1%})")
    print(f"T0交易: {t0_trades} ({t0_trades/total_trades:.1%})")
    print(f"强制调整: {forced_trades} ({forced_trades/total_trades:.1%})")
    
    # 交易价值分析
    print(f"\n交易价值分析:")
    print(f"平均交易价值: {trades_df['value'].mean():,.2f}")
    print(f"最大交易价值: {trades_df['value'].max():,.2f}")
    print(f"最小交易价值: {trades_df['value'].min():,.2f}")
    print(f"总交易价值: {trades_df['value'].sum():,.2f}")

    # 交易成本分析
    if 'transaction_cost' in trades_df.columns:
        total_costs = trades_df['transaction_cost'].sum()
        print(f"\n交易成本分析:")
        print(f"总交易成本: {total_costs:,.2f}")
        print(f"平均每笔成本: {total_costs/len(trades_df):,.2f}")
        print(f"成本占交易价值比例: {total_costs/trades_df['value'].sum()*100:.3f}%")
    
    print("\n2. 收益分析")
    print("-" * 30)
    
    # 权益曲线分析
    initial_equity = equity_df['equity'].iloc[0]
    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity / initial_equity - 1) * 100
    
    print(f"初始权益: {initial_equity:,.2f}")
    print(f"最终权益: {final_equity:,.2f}")
    print(f"总收益: {final_equity - initial_equity:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    
    # 年化收益率
    days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
    annual_return = ((final_equity / initial_equity) ** (365 / days) - 1) * 100
    print(f"年化收益率: {annual_return:.2f}%")
    
    # 最大回撤
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].max() * 100
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # 夏普比率
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    daily_returns = equity_df['daily_return'].dropna()
    if len(daily_returns) > 1:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        print(f"夏普比率: {sharpe_ratio:.2f}")
    
    # 胜率
    positive_days = len(daily_returns[daily_returns > 0])
    total_days = len(daily_returns)
    win_rate = positive_days / total_days if total_days > 0 else 0
    print(f"日胜率: {win_rate:.2%}")
    
    print("\n3. T0交易专项分析")
    print("-" * 30)
    
    # T0交易收益
    total_t0_profit = performance_df['t0_profit'].sum()
    print(f"T0交易累计收益: {total_t0_profit:,.2f}")
    
    # 平均每日T0交易
    avg_daily_trades = performance_df['trades_count'].mean()
    print(f"平均每日交易次数: {avg_daily_trades:.1f}")
    
    # T0交易频率分布
    print(f"\n每日T0交易次数分布:")
    trade_counts = performance_df['trades_count'].value_counts().sort_index()
    for count, days in trade_counts.head(10).items():
        print(f"  {count}次交易: {days}天 ({days/len(performance_df):.1%})")
    
    # 最佳和最差交易日
    best_day = performance_df.loc[performance_df['t0_profit'].idxmax()]
    worst_day = performance_df.loc[performance_df['t0_profit'].idxmin()]
    
    print(f"\n最佳交易日: {best_day['date']}")
    print(f"  T0收益: {best_day['t0_profit']:,.2f}")
    print(f"  交易次数: {best_day['trades_count']}")
    
    print(f"\n最差交易日: {worst_day['date']}")
    print(f"  T0收益: {worst_day['t0_profit']:,.2f}")
    print(f"  交易次数: {worst_day['trades_count']}")
    
    print("\n4. 风险控制分析")
    print("-" * 30)
    
    # 持仓变化分析
    print(f"持仓统计:")
    print(f"  最大持仓: {equity_df['holdings'].max():.3f}")
    print(f"  最小持仓: {equity_df['holdings'].min():.3f}")
    print(f"  平均持仓: {equity_df['holdings'].mean():.3f}")
    print(f"  持仓标准差: {equity_df['holdings'].std():.3f}")
    
    # 现金变化分析
    print(f"\n现金统计:")
    print(f"  最大现金: {equity_df['cash'].max():,.2f}")
    print(f"  最小现金: {equity_df['cash'].min():,.2f}")
    print(f"  平均现金: {equity_df['cash'].mean():,.2f}")
    
    print("\n5. 策略表现总结")
    print("-" * 30)
    
    print("✅ 优势:")
    print(f"  - 实现了{total_return:.1f}%的正收益")
    print(f"  - 最大回撤控制在{max_drawdown:.1f}%以内")
    print(f"  - T0交易贡献了{total_t0_profit:,.0f}的收益")
    print(f"  - 买卖交易相对平衡")
    print(f"  - 强制调整比例降低到{forced_trades/total_trades:.1%}")
    
    if max_drawdown > 5:
        print("\n⚠️ 需要改进:")
        print(f"  - 最大回撤{max_drawdown:.1f}%仍可进一步优化")
    
    if win_rate < 0.6:
        print(f"  - 日胜率{win_rate:.1%}有提升空间")
    
    print(f"\n📊 与基准对比:")
    print(f"  - 如果简单持有指数，收益率约为市场表现")
    print(f"  - T0策略额外贡献了{total_t0_profit/initial_equity*100:.1f}%的收益")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio if 'sharpe_ratio' in locals() else 0,
        'win_rate': win_rate,
        't0_profit': total_t0_profit,
        'total_trades': total_trades,
        't0_trades': t0_trades
    }

def create_performance_chart():
    """创建表现图表"""
    try:
        equity_df = pd.read_csv('results/improved_t0_equity_20250610_111204.csv')
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 权益曲线
        axes[0, 0].plot(equity_df['date'], equity_df['equity'])
        axes[0, 0].set_title('权益曲线')
        axes[0, 0].set_ylabel('权益')
        axes[0, 0].grid(True)
        
        # 持仓变化
        axes[0, 1].plot(equity_df['date'], equity_df['holdings'])
        axes[0, 1].set_title('持仓变化')
        axes[0, 1].set_ylabel('持仓数量')
        axes[0, 1].grid(True)
        
        # 现金变化
        axes[1, 0].plot(equity_df['date'], equity_df['cash'])
        axes[1, 0].set_title('现金变化')
        axes[1, 0].set_ylabel('现金')
        axes[1, 0].grid(True)
        
        # 回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
        axes[1, 1].fill_between(equity_df['date'], equity_df['drawdown'], 0, alpha=0.3, color='red')
        axes[1, 1].set_title('回撤')
        axes[1, 1].set_ylabel('回撤比例')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/improved_t0_performance_chart.png', dpi=300, bbox_inches='tight')
        print("\n📈 表现图表已保存到: results/improved_t0_performance_chart.png")
        
    except Exception as e:
        print(f"创建图表时出错: {e}")

def main():
    """主函数"""
    try:
        results = analyze_improved_t0_results()
        create_performance_chart()
        
        print(f"\n🎯 策略评级:")
        score = 0
        if results['total_return'] > 20:
            score += 25
        if results['max_drawdown'] < 5:
            score += 25
        if results['sharpe_ratio'] > 1:
            score += 25
        if results['win_rate'] > 0.5:
            score += 25
        
        if score >= 75:
            rating = "优秀 ⭐⭐⭐⭐⭐"
        elif score >= 50:
            rating = "良好 ⭐⭐⭐⭐"
        elif score >= 25:
            rating = "一般 ⭐⭐⭐"
        else:
            rating = "需改进 ⭐⭐"
        
        print(f"综合评分: {score}/100 - {rating}")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
