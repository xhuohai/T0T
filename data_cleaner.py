#!/usr/bin/env python3
"""
数据清理和填充脚本
处理分钟级数据的缺失问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def clean_minute_data(input_file, output_file):
    """
    清理和填充分钟级数据
    
    Args:
        input_file: str，输入文件路径
        output_file: str，输出文件路径
    """
    print(f"处理文件: {input_file}")
    
    # 读取原始数据
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()
    
    print(f"原始数据点数: {len(df)}")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 创建完整的分钟级时间序列（仅交易时间）
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    
    # 生成交易时间的完整分钟序列
    full_index = []
    current_date = start_date
    
    while current_date <= end_date:
        # 跳过周末
        if current_date.weekday() < 5:  # 0-4是周一到周五
            # 上午交易时间：9:30-11:30
            morning_start = datetime.combine(current_date, datetime.min.time().replace(hour=9, minute=30))
            morning_end = datetime.combine(current_date, datetime.min.time().replace(hour=11, minute=30))
            morning_range = pd.date_range(morning_start, morning_end, freq='1min')
            full_index.extend(morning_range)
            
            # 下午交易时间：13:00-15:00
            afternoon_start = datetime.combine(current_date, datetime.min.time().replace(hour=13, minute=0))
            afternoon_end = datetime.combine(current_date, datetime.min.time().replace(hour=15, minute=0))
            afternoon_range = pd.date_range(afternoon_start, afternoon_end, freq='1min')
            full_index.extend(afternoon_range)
        
        current_date += timedelta(days=1)
    
    full_index = pd.DatetimeIndex(full_index)
    print(f"完整时间序列长度: {len(full_index)}")
    
    # 重新索引并填充缺失数据
    df_reindexed = df.reindex(full_index)
    
    # 前向填充价格数据
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].fillna(method='ffill')
    
    # 成交量和成交额填充为0
    volume_columns = ['volume', 'amount']
    for col in volume_columns:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].fillna(0)
    
    # 删除仍然有NaN的行（通常是开头的几行）
    df_cleaned = df_reindexed.dropna()
    
    print(f"清理后数据点数: {len(df_cleaned)}")
    print(f"填充的数据点: {len(df_cleaned) - len(df)}")
    
    # 重置索引并保存
    df_cleaned = df_cleaned.reset_index()
    df_cleaned = df_cleaned.rename(columns={'index': 'datetime'})
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存清理后的数据
    df_cleaned.to_csv(output_file, index=False)
    print(f"清理后的数据已保存到: {output_file}")
    
    return df_cleaned

def validate_data_continuity(file_path):
    """
    验证数据的连续性
    
    Args:
        file_path: str，文件路径
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # 检查时间间隔
    time_diffs = df.index.to_series().diff()
    
    # 统计不同的时间间隔
    interval_counts = time_diffs.value_counts()
    
    print(f"\n数据连续性验证 - {file_path}")
    print("=" * 50)
    print("时间间隔分布:")
    for interval, count in interval_counts.head(10).items():
        if pd.notna(interval):
            print(f"  {interval}: {count} 次")
    
    # 检查是否有超过1分钟的间隔（在交易时间内）
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=1)]
    if len(large_gaps) > 0:
        print(f"\n发现 {len(large_gaps)} 个超过1分钟的间隔:")
        for gap_time, gap_duration in large_gaps.head(5).items():
            print(f"  {gap_time}: {gap_duration}")
    else:
        print("\n✅ 交易时间内数据连续，无缺失")

def main():
    """主函数"""
    print("=" * 60)
    print("数据清理和填充工具")
    print("=" * 60)
    
    # 处理所有processed目录下的文件
    processed_dir = "data/processed"
    cleaned_dir = "data/cleaned"
    
    if not os.path.exists(processed_dir):
        print(f"错误: 目录 {processed_dir} 不存在")
        return
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"错误: 在 {processed_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个文件需要处理")
    
    for csv_file in csv_files:
        input_path = os.path.join(processed_dir, csv_file)
        output_path = os.path.join(cleaned_dir, csv_file)
        
        try:
            # 清理数据
            cleaned_df = clean_minute_data(input_path, output_path)
            
            # 验证数据连续性
            validate_data_continuity(output_path)
            
            print("-" * 50)
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            continue
    
    print("\n✅ 数据清理完成！")
    print(f"清理后的数据保存在: {cleaned_dir}")

if __name__ == "__main__":
    main()
