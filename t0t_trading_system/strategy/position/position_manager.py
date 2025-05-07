"""
仓位管理模块
实现基于月线、周线、日线的仓位控制策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PositionManager:
    """仓位管理类"""
    
    def __init__(self, config):
        """
        初始化仓位管理器
        
        Args:
            config: 仓位管理配置
        """
        self.config = config
        self.max_position = config.get("max_position", 0.85)
        self.min_position = config.get("min_position", 0.15)
        self.default_position = config.get("default_position", 0.5)
        self.monthly_adjust_limit = config.get("monthly_adjust_limit", 0.10)
        self.weekly_adjust_limit = config.get("weekly_adjust_limit", 0.06)
        self.daily_adjust_limit = config.get("daily_adjust_limit", 0.04)
        self.bottom_threshold_years = config.get("bottom_threshold_years", 5)
        self.top_threshold_years = config.get("top_threshold_years", 5)
        
        # 当前仓位
        self.current_position = self.default_position
        
        # 仓位历史记录
        self.position_history = []
    
    def calculate_monthly_position(self, monthly_data, current_date=None):
        """
        计算月度仓位
        
        Args:
            monthly_data: DataFrame，月线数据，包含技术指标
            current_date: datetime，当前日期，默认为None表示使用最新数据
            
        Returns:
            float: 目标仓位
        """
        if current_date is None:
            current_date = monthly_data.index[-1]
        
        # 确保数据按日期排序
        monthly_data = monthly_data.sort_index()
        
        # 获取当前月的数据
        current_month_data = monthly_data.loc[monthly_data.index <= current_date].iloc[-1]
        
        # 获取144周期月均线
        ma144_col = 'ma_144'
        if ma144_col not in monthly_data.columns:
            raise ValueError("Monthly data must contain 144-period moving average")
        
        # 获取5周期月均线
        ma5_col = 'ma_5'
        if ma5_col not in monthly_data.columns:
            raise ValueError("Monthly data must contain 5-period moving average")
        
        # 获取过去5年的数据
        years_ago = current_date - pd.DateOffset(years=self.bottom_threshold_years)
        historical_data = monthly_data.loc[monthly_data.index >= years_ago]
        
        # 计算目标仓位
        target_position = self.current_position
        
        # 检查是否跌破144月均线（底部信号）
        if current_month_data['close'] < current_month_data[ma144_col]:
            # 检查是否是首次跌破
            prev_month_data = monthly_data.loc[monthly_data.index < current_date].iloc[-1]
            if prev_month_data['close'] >= prev_month_data[ma144_col]:
                # 首次跌破，加仓至少50%
                target_position = max(0.5, self.current_position + self.monthly_adjust_limit)
            else:
                # 持续跌破，继续加仓
                target_position = min(self.max_position, self.current_position + self.monthly_adjust_limit)
                
            # 检查是否出现底背离
            if ('bullish_divergence' in current_month_data and current_month_data['bullish_divergence'] and
                current_month_data['low'] <= historical_data['low'].min()):
                # 出现底背离且创5年新低，直接加仓至85%
                target_position = self.max_position
        
        # 检查是否出现顶部信号
        elif (current_month_data['high'] >= historical_data['high'].max() and
              'bearish_divergence' in current_month_data and current_month_data['bearish_divergence']):
            # 创5年新高且出现顶背离
            if current_month_data['close'] > current_month_data[ma5_col]:
                # 收盘价在5月均线上方，减仓
                target_position = max(self.min_position, self.current_position - self.monthly_adjust_limit)
            else:
                # 收盘价跌破5月均线，快速减仓至15%
                target_position = self.min_position
        
        # 检查是否在过去3个月内出现过5年新高，且当月收盘价跌破5月均线
        elif (monthly_data.loc[monthly_data.index >= current_date - pd.DateOffset(months=3)]['high'].max() >= 
              historical_data['high'].max() and current_month_data['close'] < current_month_data[ma5_col]):
            # 快速减仓至15%
            target_position = self.min_position
        
        # 限制仓位在允许范围内
        target_position = max(self.min_position, min(self.max_position, target_position))
        
        return target_position
    
    def calculate_weekly_position(self, weekly_data, monthly_target, current_date=None):
        """
        计算周度仓位调整
        
        Args:
            weekly_data: DataFrame，周线数据，包含技术指标
            monthly_target: float，月度目标仓位
            current_date: datetime，当前日期，默认为None表示使用最新数据
            
        Returns:
            float: 调整后的目标仓位
        """
        if current_date is None:
            current_date = weekly_data.index[-1]
        
        # 确保数据按日期排序
        weekly_data = weekly_data.sort_index()
        
        # 获取当前周的数据
        current_week_data = weekly_data.loc[weekly_data.index <= current_date].iloc[-1]
        
        # 初始化周度目标仓位为月度目标
        weekly_target = monthly_target
        
        # 检查是否出现底背离
        if 'bullish_divergence' in current_week_data and current_week_data['bullish_divergence']:
            # 出现底背离，在月度目标基础上加仓
            weekly_target = min(self.max_position, monthly_target + self.weekly_adjust_limit)
        
        # 检查是否出现顶背离
        elif 'bearish_divergence' in current_week_data and current_week_data['bearish_divergence']:
            # 出现顶背离，在月度目标基础上减仓
            weekly_target = max(self.min_position, monthly_target - self.weekly_adjust_limit)
        
        # 限制仓位在允许范围内
        weekly_target = max(self.min_position, min(self.max_position, weekly_target))
        
        return weekly_target
    
    def calculate_daily_position(self, daily_data, weekly_target, current_date=None):
        """
        计算日度仓位调整
        
        Args:
            daily_data: DataFrame，日线数据，包含技术指标
            weekly_target: float，周度目标仓位
            current_date: datetime，当前日期，默认为None表示使用最新数据
            
        Returns:
            float: 调整后的目标仓位
        """
        if current_date is None:
            current_date = daily_data.index[-1]
        
        # 确保数据按日期排序
        daily_data = daily_data.sort_index()
        
        # 获取当前日的数据
        current_day_data = daily_data.loc[daily_data.index <= current_date].iloc[-1]
        
        # 初始化日度目标仓位为周度目标
        daily_target = weekly_target
        
        # 检查是否出现底背离
        if 'bullish_divergence' in current_day_data and current_day_data['bullish_divergence']:
            # 出现底背离，在周度目标基础上加仓
            daily_target = min(self.max_position, weekly_target + self.daily_adjust_limit)
        
        # 检查是否出现顶背离
        elif 'bearish_divergence' in current_day_data and current_day_data['bearish_divergence']:
            # 出现顶背离，在周度目标基础上减仓
            daily_target = max(self.min_position, weekly_target - self.daily_adjust_limit)
        
        # 限制仓位在允许范围内
        daily_target = max(self.min_position, min(self.max_position, daily_target))
        
        return daily_target
    
    def update_position(self, monthly_data, weekly_data, daily_data, current_date=None):
        """
        更新仓位
        
        Args:
            monthly_data: DataFrame，月线数据，包含技术指标
            weekly_data: DataFrame，周线数据，包含技术指标
            daily_data: DataFrame，日线数据，包含技术指标
            current_date: datetime，当前日期，默认为None表示使用最新数据
            
        Returns:
            float: 更新后的仓位
        """
        if current_date is None:
            current_date = daily_data.index[-1]
        
        # 计算月度目标仓位
        monthly_target = self.calculate_monthly_position(monthly_data, current_date)
        
        # 计算周度目标仓位
        weekly_target = self.calculate_weekly_position(weekly_data, monthly_target, current_date)
        
        # 计算日度目标仓位
        daily_target = self.calculate_daily_position(daily_data, weekly_target, current_date)
        
        # 更新当前仓位
        self.current_position = daily_target
        
        # 记录仓位历史
        self.position_history.append({
            'date': current_date,
            'position': self.current_position,
            'monthly_target': monthly_target,
            'weekly_target': weekly_target,
            'daily_target': daily_target
        })
        
        return self.current_position
    
    def get_position_history(self):
        """
        获取仓位历史记录
        
        Returns:
            DataFrame: 仓位历史记录
        """
        return pd.DataFrame(self.position_history)
    
    def get_current_position(self):
        """
        获取当前仓位
        
        Returns:
            float: 当前仓位
        """
        return self.current_position
