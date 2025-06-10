#!/usr/bin/env python3
"""
招商银行T0策略详细分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_latest_results():
    """加载最新的回测结果"""
    import glob
    
    # 查找最新的回测结果文件
    trade_files = glob.glob("results/improved_t0_trades_*.csv")
    equity_files = glob.glob("results/improved_t0_equity_*.csv")
    
    if not trade_files or not equity_files:
        print("❌ 没有找到回测结果文件")
        return None, None
    
    # 使用最新的文件
    latest_trade_file = max(trade_files)
    latest_equity_file = max(equity_files)
    
    print(f"📁 加载交易记录: {latest_trade_file}")
    print(f"📁 加载权益曲线: {latest_equity_file}")
    
    # 读取数据
    trades_df = pd.read_csv(latest_trade_file)
    equity_df = pd.read_csv(latest_equity_file)
    
    # 转换时间列
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    return trades_df, equity_df

def analyze_trading_performance(trades_df, equity_df):
    """分析交易表现"""
    print("\n" + "="*80)
    print("📊 招商银行T0策略详细分析报告")
    print("="*80)
    
    # 基本统计
    initial_equity = equity_df['equity'].iloc[0]
    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    
    print(f"\n💰 资金表现:")
    print(f"   初始资金: {initial_equity:,.2f} 元")
    print(f"   最终资金: {final_equity:,.2f} 元")
    print(f"   总收益: {final_equity - initial_equity:,.2f} 元")
    print(f"   总收益率: {total_return:.2f}%")
    
    # 计算年化收益率
    trading_days = len(equity_df)
    years = trading_days / 252  # 假设一年252个交易日
    annual_return = (final_equity / initial_equity) ** (1/years) - 1
    print(f"   年化收益率: {annual_return*100:.2f}%")
    
    # 风险指标
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    daily_returns = equity_df['daily_return'].dropna()
    
    # 最大回撤
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax'] * 100
    max_drawdown = equity_df['drawdown'].max()
    
    # 夏普比率
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # 胜率
    positive_days = len(daily_returns[daily_returns > 0])
    win_rate = positive_days / len(daily_returns) * 100
    
    print(f"\n📈 风险指标:")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   夏普比率: {sharpe_ratio:.2f}")
    print(f"   日胜率: {win_rate:.1f}%")
    print(f"   日均收益率: {daily_returns.mean()*100:.3f}%")
    print(f"   收益波动率: {daily_returns.std()*100:.3f}%")
    
    # 交易统计
    total_trades = len(trades_df)
    buy_trades = len(trades_df[trades_df['type'] == 'buy'])
    sell_trades = len(trades_df[trades_df['type'] == 'sell'])
    
    # T0交易统计
    if 'is_t0_trade' in trades_df.columns:
        t0_trades = len(trades_df[trades_df['is_t0_trade'] == True])
    else:
        # 估算T0交易
        trades_df['date'] = trades_df['time'].dt.date
        daily_trades = trades_df.groupby('date').size()
        t0_trades = len(daily_trades[daily_trades > 1]) * 2
    
    # 强制交易
    if 'is_forced_adjustment' in trades_df.columns:
        forced_trades = len(trades_df[trades_df['is_forced_adjustment'] == True])
    else:
        forced_trades = 0
    
    print(f"\n🔄 交易统计:")
    print(f"   总交易次数: {total_trades}")
    print(f"   买入交易: {buy_trades}")
    print(f"   卖出交易: {sell_trades}")
    print(f"   T0交易: {t0_trades} ({t0_trades/total_trades*100:.1f}%)")
    print(f"   强制调整: {forced_trades}")
    print(f"   平均每日交易: {total_trades/trading_days:.1f} 次")
    
    # 交易成本分析
    if 'transaction_cost' in trades_df.columns:
        total_cost = trades_df['transaction_cost'].sum()
    else:
        total_cost = trades_df['value'].sum() * 0.0014
    
    cost_ratio = total_cost / initial_equity * 100
    
    print(f"\n💸 成本分析:")
    print(f"   总交易成本: {total_cost:,.2f} 元")
    print(f"   成本占初始资金比例: {cost_ratio:.2f}%")
    print(f"   平均每笔交易成本: {total_cost/total_trades:.2f} 元")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return * 100,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        't0_trades': t0_trades,
        'total_cost': total_cost,
        'cost_ratio': cost_ratio
    }

def create_detailed_charts(trades_df, equity_df):
    """创建详细的分析图表"""
    print(f"\n📊 生成详细分析图表...")
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('招商银行T0策略详细分析', fontsize=16, fontweight='bold')
    
    # 1. 权益曲线
    ax1 = axes[0, 0]
    ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2, color='blue')
    ax1.set_title('权益曲线')
    ax1.set_ylabel('权益 (元)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 回撤曲线
    ax2 = axes[0, 1]
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax'] * 100
    ax2.fill_between(equity_df['date'], equity_df['drawdown'], 0, alpha=0.3, color='red')
    ax2.plot(equity_df['date'], equity_df['drawdown'], color='red')
    ax2.set_title('回撤曲线')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 日收益率分布
    ax3 = axes[1, 0]
    equity_df['daily_return'] = equity_df['equity'].pct_change() * 100
    daily_returns = equity_df['daily_return'].dropna()
    ax3.hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--', label=f'均值: {daily_returns.mean():.3f}%')
    ax3.set_title('日收益率分布')
    ax3.set_xlabel('日收益率 (%)')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 月度交易次数
    ax4 = axes[1, 1]
    trades_df['month'] = trades_df['time'].dt.to_period('M')
    monthly_trades = trades_df.groupby('month').size()
    monthly_trades.plot(kind='bar', ax=ax4, color='orange', alpha=0.7)
    ax4.set_title('月度交易次数')
    ax4.set_ylabel('交易次数')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. 交易类型分布
    ax5 = axes[2, 0]
    trade_types = trades_df['type'].value_counts()
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    wedges, texts, autotexts = ax5.pie(trade_types.values, labels=trade_types.index, 
                                      autopct='%1.1f%%', colors=colors[:len(trade_types)])
    ax5.set_title('交易类型分布')
    
    # 6. 持仓变化
    ax6 = axes[2, 1]
    ax6.plot(equity_df['date'], equity_df['holdings'], linewidth=2, color='purple')
    ax6.set_title('持仓数量变化')
    ax6.set_ylabel('持仓 (股)')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = f"results/cmb_detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"📊 详细分析图表已保存到: {chart_file}")
    
    plt.show()

def generate_summary_report(stats):
    """生成汇总报告"""
    print(f"\n" + "="*80)
    print("📋 招商银行T0策略表现总结")
    print("="*80)
    
    print(f"\n🎯 核心指标:")
    print(f"   ✅ 总收益率: {stats['total_return']:.2f}%")
    print(f"   ✅ 年化收益率: {stats['annual_return']:.2f}%")
    print(f"   ✅ 最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"   ✅ 夏普比率: {stats['sharpe_ratio']:.2f}")
    print(f"   ✅ 日胜率: {stats['win_rate']:.1f}%")
    
    print(f"\n📊 交易效率:")
    print(f"   🔄 总交易次数: {stats['total_trades']}")
    print(f"   🎯 T0交易占比: {stats['t0_trades']/stats['total_trades']*100:.1f}%")
    print(f"   💰 交易成本率: {stats['cost_ratio']:.2f}%")
    
    print(f"\n🏆 策略评价:")
    if stats['total_return'] > 15:
        print("   📈 收益表现: 优秀 (>15%)")
    elif stats['total_return'] > 8:
        print("   📈 收益表现: 良好 (8-15%)")
    else:
        print("   📈 收益表现: 一般 (<8%)")
    
    if stats['max_drawdown'] < 10:
        print("   🛡️ 风险控制: 优秀 (<10%)")
    elif stats['max_drawdown'] < 20:
        print("   🛡️ 风险控制: 良好 (10-20%)")
    else:
        print("   🛡️ 风险控制: 需改进 (>20%)")
    
    if stats['sharpe_ratio'] > 1.0:
        print("   ⚖️ 风险调整收益: 优秀 (>1.0)")
    elif stats['sharpe_ratio'] > 0.5:
        print("   ⚖️ 风险调整收益: 良好 (0.5-1.0)")
    else:
        print("   ⚖️ 风险调整收益: 一般 (<0.5)")
    
    print(f"\n💡 策略建议:")
    if stats['cost_ratio'] > 15:
        print("   ⚠️ 交易成本较高，建议优化交易频率")
    if stats['max_drawdown'] > 15:
        print("   ⚠️ 最大回撤较大，建议加强风险控制")
    if stats['t0_trades']/stats['total_trades'] < 0.6:
        print("   ⚠️ T0交易占比偏低，建议优化T0策略")
    
    print(f"\n✅ 总体评价: 招商银行T0策略表现{('优秀' if stats['total_return'] > 15 and stats['max_drawdown'] < 15 else '良好')}")

def main():
    """主函数"""
    print("🚀 招商银行T0策略详细分析")
    print("="*60)
    
    # 加载数据
    trades_df, equity_df = load_latest_results()
    if trades_df is None or equity_df is None:
        return
    
    # 分析表现
    stats = analyze_trading_performance(trades_df, equity_df)
    
    # 创建图表
    create_detailed_charts(trades_df, equity_df)
    
    # 生成总结报告
    generate_summary_report(stats)
    
    print(f"\n🎉 招商银行T0策略分析完成！")

if __name__ == "__main__":
    main()
