"""
全局配置文件
"""

# 数据相关配置
DATA_CONFIG = {
    "index_symbol": "000001.SH",  # 上证指数
    "data_source": "tushare",     # 数据源
    "data_dir": "data/storage",   # 数据存储目录
    "cache_data": True,           # 是否缓存数据
}

# 技术指标配置
INDICATOR_CONFIG = {
    # MACD参数
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    # KDJ参数
    "kdj": {
        "k_period": 9,
        "d_period": 3,
        "j_period": 3
    },
    # 均线参数
    "ma": {
        "monthly_periods": [5, 10, 20, 60, 144],  # 月均线周期
        "weekly_periods": [5, 10, 20, 60],        # 周均线周期
        "daily_periods": [5, 10, 20, 60, 120],    # 日均线周期
    },
    # ATR参数
    "atr": {
        "period": 14
    }
}

# 仓位管理配置
POSITION_CONFIG = {
    "max_position": 0.85,  # 最大仓位
    "min_position": 0.15,  # 最小仓位
    "default_position": 0.5,  # 默认仓位
    "monthly_adjust_limit": 0.10,  # 月度调整上限
    "weekly_adjust_limit": 0.06,   # 周度调整上限
    "daily_adjust_limit": 0.04,    # 日度调整上限
    "bottom_threshold_years": 5,   # 底部判断参考年数
    "top_threshold_years": 5,      # 顶部判断参考年数
}

# T0交易配置
T0_CONFIG = {
    "min_trade_portion": 1/8,  # 最小交易份额
    "max_trade_portion": 1/3,  # 最大交易份额
    "fib_tolerance": 0.005,    # 斐波那契位置容差
    "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.382, 1.618],  # 斐波那契水平
    "price_tolerance": 0.005,  # 价格容差
}

# 风险管理配置
RISK_CONFIG = {
    "stop_loss_pct": 0.02,  # 止损百分比
    "max_drawdown": 0.1,    # 最大回撤限制
}

# 回测配置
BACKTEST_CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 1000000,  # 初始资金
    "commission_rate": 0.0003,   # 佣金率
    "slippage": 0.0001,          # 滑点
    "tax_rate": 0.001,           # 印花税率
}
