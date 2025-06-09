"""
本地数据源模块
用于读取本地CSV文件作为数据源
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LocalDataSource:
    """本地数据源类"""
    
    def __init__(self, data_dir="data/processed"):
        """
        初始化本地数据源
        
        Args:
            data_dir: 本地数据目录
        """
        self.data_dir = data_dir
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
            
        logger.info(f"初始化本地数据源，数据目录: {data_dir}")
    
    def get_available_symbols(self):
        """
        获取可用的股票代码列表
        
        Returns:
            list: 可用的股票代码列表
        """
        symbols = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '')
                symbols.append(symbol)
        return symbols
    
    def load_stock_data(self, symbol, start_date=None, end_date=None):
        """
        加载股票数据
        
        Args:
            symbol: 股票代码，如 'SH600000'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: 股票数据
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"数据文件不存在: {file_path}")
            return None
            
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 转换时间列
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # 确保列名标准化
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount'
            })
            
            # 过滤日期范围
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]
            
            # 按时间排序
            df = df.sort_index()
            
            logger.info(f"成功加载 {symbol} 数据，共 {len(df)} 条记录")
            logger.info(f"时间范围: {df.index.min()} 到 {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"加载数据文件失败 {file_path}: {e}")
            return None
    
    def convert_to_daily(self, df):
        """
        将分钟数据转换为日线数据
        
        Args:
            df: 分钟级数据
            
        Returns:
            pandas.DataFrame: 日线数据
        """
        if df is None or df.empty:
            return None
            
        # 按日期分组，计算OHLC
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        # 删除没有交易的日期
        daily_df = daily_df.dropna()
        
        return daily_df
    
    def convert_to_weekly(self, df):
        """
        将日线数据转换为周线数据
        
        Args:
            df: 日线数据
            
        Returns:
            pandas.DataFrame: 周线数据
        """
        if df is None or df.empty:
            return None
            
        # 按周分组，计算OHLC
        weekly_df = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        # 删除没有交易的周
        weekly_df = weekly_df.dropna()
        
        return weekly_df
    
    def convert_to_monthly(self, df):
        """
        将日线数据转换为月线数据
        
        Args:
            df: 日线数据
            
        Returns:
            pandas.DataFrame: 月线数据
        """
        if df is None or df.empty:
            return None
            
        # 按月分组，计算OHLC
        monthly_df = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        # 删除没有交易的月
        monthly_df = monthly_df.dropna()
        
        return monthly_df
    
    def get_multi_timeframe_data(self, symbol, start_date=None, end_date=None):
        """
        获取多时间周期数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 包含不同时间周期的数据
        """
        # 加载分钟数据
        minute_data = self.load_stock_data(symbol, start_date, end_date)
        
        if minute_data is None:
            return None
            
        # 转换为不同时间周期
        daily_data = self.convert_to_daily(minute_data)
        weekly_data = self.convert_to_weekly(daily_data)
        monthly_data = self.convert_to_monthly(daily_data)
        
        return {
            'minute': minute_data,
            'daily': daily_data,
            'weekly': weekly_data,
            'monthly': monthly_data
        }
