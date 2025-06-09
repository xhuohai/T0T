#!/usr/bin/env python3
"""
数据处理脚本
用于清洗和整理A股分钟级交易数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, time
import glob
from tqdm import tqdm
import re

class DataProcessor:
    def __init__(self, source_dir, target_dir):
        """
        初始化数据处理器
        
        Args:
            source_dir: 源数据目录
            target_dir: 目标数据目录
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        
        # 选择的10支常见标的
        self.selected_stocks = [
            'SH000001',  # 上证指数
            'SH600000',  # 浦发银行
            'SZ000001',  # 平安银行
            'SZ000002',  # 万科A
            'SH600028',  # 中国石化
            'SH600036',  # 招商银行
            'SH600519',  # 贵州茅台
            'SH601318',  # 中国平安
            'SH601398',  # 工商银行
            'SZ002594',  # 比亚迪
        ]
        
        # 交易时间范围
        self.trading_start = time(9, 30)  # 9:30
        self.trading_end = time(15, 0)    # 15:00
        
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
    def parse_ohlc_field(self, field_value):
        """
        解析合并的OHLC字段
        例如: "7.2707.280" -> (7.270, 7.280)
        """
        if pd.isna(field_value) or field_value == '':
            return None, None

        field_str = str(field_value)

        # 尝试多种模式匹配
        patterns = [
            r'(\d+\.\d{3})(\d+\.\d{3})',  # 例如: 7.2707.280
            r'(\d+\.\d{2})(\d+\.\d{2})',  # 例如: 7.277.28
            r'(\d+\.\d+)(\d+\.\d+)',      # 通用模式
        ]

        for pattern in patterns:
            match = re.match(pattern, field_str)
            if match:
                val1 = float(match.group(1))
                val2 = float(match.group(2))
                return val1, val2

        # 如果都匹配失败，尝试单个数值
        try:
            value = float(field_str)
            return value, value
        except:
            return None, None
    
    def clean_single_file(self, file_path):
        """
        清洗单个CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame: 清洗后的数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查数据是否为空
            if df.empty:
                print(f"警告: {file_path} 文件为空")
                return None
                
            # 解析时间列
            df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
            
            # 解析OHLC字段
            open_vals = []
            high_vals = []
            low_vals = []
            close_vals = []
            volume_vals = []
            amount_vals = []

            for _, row in df.iterrows():
                # 解析Open字段 (开盘价和收盘价)
                open_price, close_price = self.parse_ohlc_field(row['Open'])
                # 解析High字段 (最高价和最低价)
                high_price, low_price = self.parse_ohlc_field(row['High'])
                # 解析Low字段 (成交量和成交额)
                volume, amount = self.parse_ohlc_field(row['Low'])

                # 根据数据格式，重新分配字段
                open_vals.append(open_price if open_price is not None else 0)
                close_vals.append(close_price if close_price is not None else 0)
                high_vals.append(high_price if high_price is not None else 0)
                low_vals.append(low_price if low_price is not None else 0)
                volume_vals.append(volume if volume is not None else 0)
                amount_vals.append(amount if amount is not None else 0)
            
            # 创建新的DataFrame
            cleaned_df = pd.DataFrame({
                'datetime': df['Time'],
                'open': open_vals,
                'high': high_vals,
                'low': low_vals,
                'close': close_vals,
                'volume': volume_vals,
                'amount': amount_vals
            })
            
            # 过滤交易时间
            cleaned_df = self.filter_trading_hours(cleaned_df)
            
            # 处理缺失值
            cleaned_df = self.handle_missing_values(cleaned_df)
            
            # 数据验证
            cleaned_df = self.validate_data(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    def filter_trading_hours(self, df):
        """
        过滤交易时间，只保留9:30-15:00的数据
        """
        df = df.copy()
        df['time'] = df['datetime'].dt.time
        
        # 过滤交易时间
        mask = (df['time'] >= self.trading_start) & (df['time'] <= self.trading_end)
        filtered_df = df[mask].copy()
        
        # 删除临时列
        filtered_df = filtered_df.drop('time', axis=1)
        
        return filtered_df
    
    def handle_missing_values(self, df):
        """
        处理缺失值
        """
        df = df.copy()
        
        # 对于价格数据，使用前向填充
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # 对于成交量，使用0填充
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0)
        
        # 删除仍然有缺失值的行
        df = df.dropna()
        
        return df
    
    def validate_data(self, df):
        """
        数据验证和清理
        """
        df = df.copy()
        
        # 确保价格数据为正数
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # 确保high >= low
        if 'high' in df.columns and 'low' in df.columns:
            df = df[df['high'] >= df['low']]
        
        # 确保成交量为非负数
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        if 'amount' in df.columns:
            df = df[df['amount'] >= 0]
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def process_stock_data(self, stock_code):
        """
        处理单支股票的所有数据
        
        Args:
            stock_code: 股票代码，如 'SH600000'
        """
        print(f"处理股票: {stock_code}")
        
        # 查找该股票的所有数据文件
        pattern = f"{self.source_dir}/**/{stock_code}.csv"
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print(f"未找到股票 {stock_code} 的数据文件")
            return
        
        all_data = []
        
        # 处理每个文件
        for file_path in tqdm(files, desc=f"处理 {stock_code}"):
            cleaned_data = self.clean_single_file(file_path)
            if cleaned_data is not None and not cleaned_data.empty:
                all_data.append(cleaned_data)
        
        if not all_data:
            print(f"股票 {stock_code} 没有有效数据")
            return
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 去重并排序
        combined_data = combined_data.drop_duplicates(subset=['datetime'])
        combined_data = combined_data.sort_values('datetime').reset_index(drop=True)
        
        # 保存到目标目录
        output_file = os.path.join(self.target_dir, f"{stock_code}.csv")
        combined_data.to_csv(output_file, index=False)
        
        print(f"股票 {stock_code} 数据处理完成，共 {len(combined_data)} 条记录")
        print(f"时间范围: {combined_data['datetime'].min()} 到 {combined_data['datetime'].max()}")
        print(f"保存到: {output_file}")
        print("-" * 50)
    
    def process_all_stocks(self):
        """
        处理所有选定的股票数据
        """
        print("开始处理股票数据...")
        print(f"源数据目录: {self.source_dir}")
        print(f"目标数据目录: {self.target_dir}")
        print(f"选定股票: {self.selected_stocks}")
        print("=" * 50)
        
        for stock_code in self.selected_stocks:
            try:
                self.process_stock_data(stock_code)
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {e}")
                continue
        
        print("所有股票数据处理完成！")

def main():
    """主函数"""
    source_dir = "/home/chenghai/Work/LLM/data/extracted/"
    target_dir = "./data/processed/"
    
    processor = DataProcessor(source_dir, target_dir)
    processor.process_all_stocks()

if __name__ == "__main__":
    main()
