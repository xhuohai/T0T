#!/usr/bin/env python3
"""
T0交易系统可交互展示应用
使用Streamlit创建可交互的回测结果展示
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# 设置页面配置
st.set_page_config(
    page_title="T0交易系统回测分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_backtest_results():
    """加载回测结果数据"""
    results_dir = "results"
    
    # 查找最新的结果文件
    files = os.listdir(results_dir) if os.path.exists(results_dir) else []
    
    # 查找最新的交易记录文件
    trade_files = [f for f in files if f.startswith('improved_t0_trades_') and f.endswith('.csv')]
    equity_files = [f for f in files if f.startswith('improved_t0_equity_') and f.endswith('.csv')]
    
    if not trade_files or not equity_files:
        return None, None, None
    
    # 使用最新的文件
    latest_trade_file = sorted(trade_files)[-1]
    latest_equity_file = sorted(equity_files)[-1]
    
    # 加载数据
    trades_df = pd.read_csv(os.path.join(results_dir, latest_trade_file))
    equity_df = pd.read_csv(os.path.join(results_dir, latest_equity_file))
    
    # 转换时间列
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    return trades_df, equity_df, latest_trade_file

@st.cache_data
def load_price_data():
    """加载价格数据"""
    # 使用清理后的数据
    data_file = "data/cleaned/SH000001.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    return None

def create_price_chart_with_signals(price_data, trades_data, date_range):
    """创建带有交易信号的价格图表"""
    # 过滤数据到指定日期范围
    start_date, end_date = date_range

    price_filtered = price_data[
        (price_data['datetime'].dt.date >= start_date) &
        (price_data['datetime'].dt.date <= end_date)
    ].copy()

    trades_filtered = trades_data[
        (trades_data['time'].dt.date >= start_date) &
        (trades_data['time'].dt.date <= end_date)
    ].copy()

    if price_filtered.empty:
        return None

    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('价格走势与交易信号', '成交量', '持仓变化', '现金流变化'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # 价格K线图
    fig.add_trace(
        go.Candlestick(
            x=price_filtered['datetime'],
            open=price_filtered['open'],
            high=price_filtered['high'],
            low=price_filtered['low'],
            close=price_filtered['close'],
            name='上证指数',
            increasing_line_color='red',
            decreasing_line_color='green'
        ),
        row=1, col=1
    )

    # 添加价格移动平均线
    if len(price_filtered) > 20:
        price_filtered['ma20'] = price_filtered['close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=price_filtered['datetime'],
                y=price_filtered['ma20'],
                mode='lines',
                name='MA20',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )

    if len(price_filtered) > 60:
        price_filtered['ma60'] = price_filtered['close'].rolling(window=60).mean()
        fig.add_trace(
            go.Scatter(
                x=price_filtered['datetime'],
                y=price_filtered['ma60'],
                mode='lines',
                name='MA60',
                line=dict(color='orange', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # 添加买入信号
    buy_trades = trades_filtered[trades_filtered['type'] == 'buy']
    if not buy_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_trades['time'],
                y=buy_trades['price'],  # 使用交易记录中的价格
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='买入信号',
                text=[f"买入: {row['volume']:.2f}股<br>价格: {row['price']:.2f}"
                      for _, row in buy_trades.iterrows()],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 添加卖出信号
    sell_trades = trades_filtered[trades_filtered['type'] == 'sell']
    if not sell_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_trades['time'],
                y=sell_trades['price'],  # 使用交易记录中的价格
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='卖出信号',
                text=[f"卖出: {row['volume']:.2f}股<br>价格: {row['price']:.2f}"
                      for _, row in sell_trades.iterrows()],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 成交量
    fig.add_trace(
        go.Bar(
            x=price_filtered['datetime'],
            y=price_filtered['volume'],
            name='成交量',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 持仓变化
    if not trades_filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_filtered['time'],
                y=trades_filtered['holdings_after'],
                mode='lines+markers',
                name='持仓数量',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )

    # 现金流变化
    if not trades_filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_filtered['time'],
                y=trades_filtered['cash_after'],
                mode='lines+markers',
                name='现金余额',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ),
            row=4, col=1
        )

        # 添加交易成本累计
        if 'transaction_cost' in trades_filtered.columns:
            cumulative_costs = trades_filtered['transaction_cost'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=trades_filtered['time'],
                    y=cumulative_costs,
                    mode='lines',
                    name='累计交易成本',
                    line=dict(color='red', width=1, dash='dash'),
                    yaxis='y5'
                ),
                row=4, col=1
            )
    
    # 更新布局
    fig.update_layout(
        title=f"T0交易系统回测分析 ({start_date} 至 {end_date})",
        xaxis_title="时间",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # 更新y轴标签
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="持仓", row=3, col=1)
    fig.update_yaxes(title_text="现金", row=4, col=1)
    
    return fig

def create_equity_curve_chart(equity_data):
    """创建权益曲线图表"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('权益曲线', '回撤'),
        row_heights=[0.7, 0.3]
    )
    
    # 权益曲线
    fig.add_trace(
        go.Scatter(
            x=equity_data['date'],
            y=equity_data['equity'],
            mode='lines',
            name='总权益',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 计算回撤
    equity_data['cummax'] = equity_data['equity'].cummax()
    equity_data['drawdown'] = (equity_data['cummax'] - equity_data['equity']) / equity_data['cummax']
    
    # 回撤图
    fig.add_trace(
        go.Scatter(
            x=equity_data['date'],
            y=-equity_data['drawdown'] * 100,  # 转换为百分比并取负值
            mode='lines',
            name='回撤 (%)',
            fill='tonexty',
            line=dict(color='red', width=1),
            fillcolor='rgba(255,0,0,0.3)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="权益曲线与回撤分析",
        height=600,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="权益", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    
    return fig

def calculate_performance_metrics(equity_data, trades_data):
    """计算性能指标"""
    if equity_data.empty:
        return {}
    
    initial_equity = equity_data['equity'].iloc[0]
    final_equity = equity_data['equity'].iloc[-1]
    
    # 基本收益指标
    total_return = (final_equity / initial_equity - 1) * 100
    
    # 计算年化收益率
    days = (equity_data['date'].iloc[-1] - equity_data['date'].iloc[0]).days
    annual_return = ((final_equity / initial_equity) ** (365 / days) - 1) * 100 if days > 0 else 0
    
    # 最大回撤
    equity_data['cummax'] = equity_data['equity'].cummax()
    equity_data['drawdown'] = (equity_data['cummax'] - equity_data['equity']) / equity_data['cummax']
    max_drawdown = equity_data['drawdown'].max() * 100
    
    # 夏普比率
    equity_data['daily_return'] = equity_data['equity'].pct_change()
    daily_returns = equity_data['daily_return'].dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # 胜率
    positive_days = len(daily_returns[daily_returns > 0])
    total_days = len(daily_returns)
    win_rate = positive_days / total_days * 100 if total_days > 0 else 0
    
    # 交易统计
    total_trades = len(trades_data)
    buy_trades = len(trades_data[trades_data['type'] == 'buy'])
    sell_trades = len(trades_data[trades_data['type'] == 'sell'])
    t0_trades = len(trades_data[trades_data.get('is_t0_trade', False) == True])
    forced_trades = len(trades_data[trades_data.get('is_forced_adjustment', False) == True])

    # 交易成本统计
    total_transaction_costs = 0
    if 'transaction_cost' in trades_data.columns:
        total_transaction_costs = trades_data['transaction_cost'].sum()

    # 净收益计算（扣除交易成本）
    gross_return = total_return
    net_return = gross_return - (total_transaction_costs / initial_equity * 100)
    
    return {
        'total_return': total_return,
        'net_return': net_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        't0_trades': t0_trades,
        'forced_trades': forced_trades,
        'total_transaction_costs': total_transaction_costs,
        'initial_equity': initial_equity,
        'final_equity': final_equity
    }

def main():
    """主函数"""
    st.title("📈 T0交易系统回测分析")
    st.markdown("---")
    
    # 加载数据
    with st.spinner("加载回测数据..."):
        trades_data, equity_data, result_file = load_backtest_results()
        price_data = load_price_data()
    
    if trades_data is None or equity_data is None:
        st.error("❌ 未找到回测结果文件，请先运行回测！")
        st.info("请运行 `python run_t0_backtest.py` 生成回测结果")
        return
    
    if price_data is None:
        st.error("❌ 未找到价格数据文件！")
        return
    
    # 侧边栏控制
    st.sidebar.header("📊 分析控制")
    
    # 日期范围选择
    min_date = trades_data['time'].dt.date.min()
    max_date = trades_data['time'].dt.date.max()
    
    date_range = st.sidebar.date_input(
        "选择分析日期范围",
        value=(min_date, min_date + timedelta(days=7)),  # 默认显示一周
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) != 2:
        st.sidebar.warning("请选择完整的日期范围")
        return
    
    # 计算性能指标
    metrics = calculate_performance_metrics(equity_data, trades_data)
    
    # 显示关键指标
    st.header("🎯 关键性能指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="总收益率",
            value=f"{metrics['total_return']:.2f}%",
            delta=f"净收益: {metrics['net_return']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="最大回撤",
            value=f"{metrics['max_drawdown']:.2f}%",
            delta=f"夏普比率: {metrics['sharpe_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            label="日胜率",
            value=f"{metrics['win_rate']:.1f}%",
            delta=f"总交易: {metrics['total_trades']}"
        )
    
    with col4:
        st.metric(
            label="交易成本",
            value=f"{metrics['total_transaction_costs']:,.0f}",
            delta=f"占比: {metrics['total_transaction_costs']/metrics['initial_equity']*100:.2f}%"
        )
    
    # 权益曲线
    st.header("📈 权益曲线分析")
    equity_chart = create_equity_curve_chart(equity_data)
    st.plotly_chart(equity_chart, use_container_width=True)
    
    # 价格图表与交易信号
    st.header("🎯 价格走势与交易信号")
    
    if len(date_range) == 2:
        price_chart = create_price_chart_with_signals(price_data, trades_data, date_range)
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)
        else:
            st.warning("所选日期范围内没有数据")
    
    # 详细统计
    st.header("📊 详细统计信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("交易统计")
        st.write(f"- 总交易次数: {metrics['total_trades']}")
        st.write(f"- 买入交易: {metrics['buy_trades']} ({metrics['buy_trades']/metrics['total_trades']*100:.1f}%)")
        st.write(f"- 卖出交易: {metrics['sell_trades']} ({metrics['sell_trades']/metrics['total_trades']*100:.1f}%)")
        st.write(f"- T0交易: {metrics['t0_trades']} ({metrics['t0_trades']/metrics['total_trades']*100:.1f}%)")
        st.write(f"- 强制调整: {metrics['forced_trades']} ({metrics['forced_trades']/metrics['total_trades']*100:.1f}%)")
    
    with col2:
        st.subheader("收益统计")
        st.write(f"- 初始权益: {metrics['initial_equity']:,.2f}")
        st.write(f"- 最终权益: {metrics['final_equity']:,.2f}")
        st.write(f"- 绝对收益: {metrics['final_equity'] - metrics['initial_equity']:,.2f}")
        st.write(f"- 总收益率: {metrics['total_return']:.2f}%")
        st.write(f"- 净收益率: {metrics['net_return']:.2f}%")
        st.write(f"- 年化收益率: {metrics['annual_return']:.2f}%")
        st.write(f"- 交易成本: {metrics['total_transaction_costs']:,.2f}")
        st.write(f"- 成本占比: {metrics['total_transaction_costs']/metrics['initial_equity']*100:.2f}%")
    
    # 交易记录表
    st.header("📋 交易记录")
    
    # 过滤交易记录
    if len(date_range) == 2:
        filtered_trades = trades_data[
            (trades_data['time'].dt.date >= date_range[0]) & 
            (trades_data['time'].dt.date <= date_range[1])
        ].copy()
    else:
        filtered_trades = trades_data.copy()
    
    # 显示交易记录
    if not filtered_trades.empty:
        # 选择要显示的列
        display_columns = ['time', 'type', 'price', 'volume', 'value']
        if 'transaction_cost' in filtered_trades.columns:
            display_columns.append('transaction_cost')
        if 'net_value' in filtered_trades.columns:
            display_columns.append('net_value')
        display_columns.append('holdings_after')

        # 格式化显示
        display_trades = filtered_trades[display_columns].copy()
        display_trades['time'] = display_trades['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_trades['price'] = display_trades['price'].round(2)
        display_trades['volume'] = display_trades['volume'].round(3)
        display_trades['value'] = display_trades['value'].round(2)
        if 'transaction_cost' in display_trades.columns:
            display_trades['transaction_cost'] = display_trades['transaction_cost'].round(2)
        if 'net_value' in display_trades.columns:
            display_trades['net_value'] = display_trades['net_value'].round(2)
        display_trades['holdings_after'] = display_trades['holdings_after'].round(3)
        
        # 动态构建列配置
        column_config = {
            'time': '时间',
            'type': '类型',
            'price': '价格',
            'volume': '数量',
            'value': '金额',
            'holdings_after': '持仓'
        }
        if 'transaction_cost' in display_trades.columns:
            column_config['transaction_cost'] = '交易成本'
        if 'net_value' in display_trades.columns:
            column_config['net_value'] = '净值变化'

        st.dataframe(
            display_trades,
            column_config=column_config,
            use_container_width=True
        )
    else:
        st.info("所选日期范围内没有交易记录")
    
    # 文件信息
    st.sidebar.markdown("---")
    st.sidebar.info(f"📁 数据文件: {result_file}")
    st.sidebar.info(f"📅 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
