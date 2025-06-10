#!/usr/bin/env python3
"""
多股票T0策略对比分析
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher
from t0t_trading_system.strategy.t0.t0_trader import ImprovedT0Trader
from t0t_trading_system.strategy.technical.indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_single_stock_backtest(symbol, config):
    """运行单支股票的回测 - 简化版本，直接读取已有的回测结果"""
    print(f"\n{'='*60}")
    print(f"分析 {symbol} 的回测结果")
    print(f"{'='*60}")

    try:
        # 查找最新的回测结果文件
        results_dir = "results"
        if not os.path.exists(results_dir):
            print(f"❌ {symbol} 结果目录不存在")
            return None

        # 如果是招商银行，使用已有的回测结果
        if symbol == 'SH600036':
            # 查找最新的回测结果
            import glob
            trade_files = glob.glob(f"{results_dir}/improved_t0_trades_*.csv")
            equity_files = glob.glob(f"{results_dir}/improved_t0_equity_*.csv")

            if not trade_files or not equity_files:
                print(f"❌ {symbol} 没有找到回测结果文件")
                return None

            # 使用最新的文件
            latest_trade_file = max(trade_files)
            latest_equity_file = max(equity_files)

            print(f"使用回测结果文件: {latest_trade_file}")

            # 读取数据
            trades_df = pd.read_csv(latest_trade_file)
            equity_df = pd.read_csv(latest_equity_file)

            # 转换时间列
            trades_df['time'] = pd.to_datetime(trades_df['time'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])

        else:
            # 对于其他股票，我们需要运行回测
            print(f"⚠️ {symbol} 需要单独运行回测，当前仅支持招商银行(SH600036)")
            return None

        if len(trades_df) == 0:
            print(f"❌ {symbol} 没有交易记录")
            return None

        # 计算关键指标
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # 计算最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].max()

        # 计算夏普比率
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        if len(daily_returns) > 1:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # 交易统计
        buy_trades = len(trades_df[trades_df['type'] == 'buy'])
        sell_trades = len(trades_df[trades_df['type'] == 'sell'])

        # 检查是否有T0交易标记
        if 'is_t0_trade' in trades_df.columns:
            t0_trades = len(trades_df[trades_df['is_t0_trade'] == True])
        else:
            # 估算T0交易（同日买卖）
            trades_df['date'] = trades_df['time'].dt.date
            daily_trades = trades_df.groupby('date').size()
            t0_trades = len(daily_trades[daily_trades > 1]) * 2  # 估算

        # 强制交易
        if 'is_forced_adjustment' in trades_df.columns:
            forced_trades = len(trades_df[trades_df['is_forced_adjustment'] == True])
        else:
            forced_trades = 0

        # 交易成本
        if 'transaction_cost' in trades_df.columns:
            total_cost = trades_df['transaction_cost'].sum()
        else:
            # 估算交易成本
            total_cost = trades_df['value'].sum() * 0.0014

        performance = {
            'symbol': symbol,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades_df),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            't0_trades': t0_trades,
            'forced_trades': forced_trades,
            'transaction_cost': total_cost,
            'cost_ratio': total_cost / initial_equity * 100,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'data_points': len(equity_df),
            'trading_days': len(equity_df)
        }

        print(f"✅ {symbol} 分析完成:")
        print(f"   总收益率: {total_return:.2f}%")
        print(f"   最大回撤: {max_drawdown:.2f}%")
        print(f"   夏普比率: {sharpe_ratio:.2f}")
        print(f"   总交易次数: {len(trades_df)}")
        print(f"   T0交易数量: {t0_trades}")

        return performance

    except Exception as e:
        print(f"❌ {symbol} 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison_charts(results_df):
    """创建对比图表"""
    if results_df.empty:
        print("没有有效的回测结果")
        return
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('多股票T0策略表现对比分析', fontsize=16, fontweight='bold')
    
    # 1. 总收益率对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(results_df['symbol'], results_df['total_return'], 
                   color=['green' if x > 0 else 'red' for x in results_df['total_return']])
    ax1.set_title('总收益率对比 (%)')
    ax1.set_ylabel('收益率 (%)')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['total_return']):
        ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. 最大回撤对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(results_df['symbol'], results_df['max_drawdown'], 
                   color='orange', alpha=0.7)
    ax2.set_title('最大回撤对比 (%)')
    ax2.set_ylabel('回撤 (%)')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['max_drawdown']):
        ax2.text(i, v - 0.5, f'{v:.1f}%', ha='center', va='top')
    
    # 3. 夏普比率对比
    ax3 = axes[0, 2]
    bars3 = ax3.bar(results_df['symbol'], results_df['sharpe_ratio'], 
                   color='blue', alpha=0.7)
    ax3.set_title('夏普比率对比')
    ax3.set_ylabel('夏普比率')
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['sharpe_ratio']):
        ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # 4. 交易次数对比
    ax4 = axes[1, 0]
    bars4 = ax4.bar(results_df['symbol'], results_df['total_trades'], 
                   color='purple', alpha=0.7)
    ax4.set_title('总交易次数对比')
    ax4.set_ylabel('交易次数')
    ax4.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['total_trades']):
        ax4.text(i, v + 10, f'{v}', ha='center', va='bottom')
    
    # 5. T0交易占比
    ax5 = axes[1, 1]
    t0_ratio = results_df['t0_trades'] / results_df['total_trades'] * 100
    bars5 = ax5.bar(results_df['symbol'], t0_ratio, 
                   color='cyan', alpha=0.7)
    ax5.set_title('T0交易占比 (%)')
    ax5.set_ylabel('T0占比 (%)')
    ax5.tick_params(axis='x', rotation=45)
    for i, v in enumerate(t0_ratio):
        ax5.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 6. 交易成本占比
    ax6 = axes[1, 2]
    bars6 = ax6.bar(results_df['symbol'], results_df['cost_ratio'], 
                   color='red', alpha=0.7)
    ax6.set_title('交易成本占比 (%)')
    ax6.set_ylabel('成本占比 (%)')
    ax6.tick_params(axis='x', rotation=45)
    for i, v in enumerate(results_df['cost_ratio']):
        ax6.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = f"results/multi_stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存到: {chart_file}")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 多股票T0策略对比分析")
    print("="*80)
    
    # 配置参数
    config = {
        'data': {
            'source': 'local',
            'local_data_dir': 'data/fixed_processed'
        },
        'backtest': {
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 1000000,
            'base_position_ratio': 0.5,
            'transaction_cost_rate': 0.0014
        },
        'strategy': {
            'ma_short': 20,
            'ma_long': 60,
            'volatility_window': 20,
            'volatility_threshold': 0.02,
            'min_trade_interval': 5,
            'max_consecutive_trades': 2,
            'min_price_change': 0.003
        }
    }
    
    # 要分析的股票列表（当前只支持招商银行，因为只有它有完整的回测结果）
    symbols = [
        'SH600036',  # 招商银行 (已有完整回测结果)
    ]

    print("📝 注意：当前版本只分析招商银行(SH600036)的回测结果")
    print("     其他股票需要先运行完整的回测流程")
    
    # 运行回测
    results = []
    for symbol in symbols:
        result = run_single_stock_backtest(symbol, config)
        if result:
            results.append(result)
    
    if not results:
        print("❌ 没有成功的回测结果")
        return
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_file = f"results/multi_stock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n📁 详细结果已保存到: {results_file}")
    
    # 显示汇总表
    print(f"\n📊 多股票T0策略表现汇总")
    print("="*100)
    print(f"{'股票代码':<12} {'总收益率':<10} {'最大回撤':<10} {'夏普比率':<10} {'交易次数':<10} {'T0占比':<10} {'成本占比':<10}")
    print("-"*100)
    
    for _, row in results_df.iterrows():
        t0_ratio = row['t0_trades'] / row['total_trades'] * 100
        print(f"{row['symbol']:<12} {row['total_return']:<10.2f} {row['max_drawdown']:<10.2f} "
              f"{row['sharpe_ratio']:<10.2f} {row['total_trades']:<10} {t0_ratio:<10.1f} {row['cost_ratio']:<10.2f}")
    
    # 创建对比图表
    print(f"\n📈 生成对比图表...")
    create_comparison_charts(results_df)
    
    # 排名分析
    print(f"\n🏆 策略表现排名")
    print("="*50)
    
    # 按总收益率排名
    top_return = results_df.nlargest(3, 'total_return')
    print("📈 收益率前三名:")
    for i, (_, row) in enumerate(top_return.iterrows(), 1):
        print(f"  {i}. {row['symbol']}: {row['total_return']:.2f}%")
    
    # 按夏普比率排名
    top_sharpe = results_df.nlargest(3, 'sharpe_ratio')
    print("\n📊 夏普比率前三名:")
    for i, (_, row) in enumerate(top_sharpe.iterrows(), 1):
        print(f"  {i}. {row['symbol']}: {row['sharpe_ratio']:.2f}")
    
    # 最小回撤
    min_drawdown = results_df.nsmallest(3, 'max_drawdown')
    print("\n🛡️ 最小回撤前三名:")
    for i, (_, row) in enumerate(min_drawdown.iterrows(), 1):
        print(f"  {i}. {row['symbol']}: {row['max_drawdown']:.2f}%")
    
    print(f"\n✅ 多股票对比分析完成！")

if __name__ == "__main__":
    main()
