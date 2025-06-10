#!/usr/bin/env python3
"""
测试数据解析脚本
"""

import pandas as pd
import re

def parse_ohlc_field(field_value):
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

# 测试数据解析
test_file = "/home/chenghai/Work/LLM/data/extracted/2023/01/2023-01-03/SH600000.csv"

print("读取测试文件...")
df = pd.read_csv(test_file)
print(f"原始数据形状: {df.shape}")
print("\n原始数据前5行:")
print(df.head())

print("\n解析字段...")
for i in range(min(5, len(df))):
    row = df.iloc[i]
    print(f"\n第{i+1}行:")
    print(f"  Time: {row['Time']}")
    print(f"  Open: {row['Open']} -> {parse_ohlc_field(row['Open'])}")
    print(f"  High: {row['High']} -> {parse_ohlc_field(row['High'])}")
    print(f"  Low: {row['Low']} -> {parse_ohlc_field(row['Low'])}")
    print(f"  Close: {row['Close']} -> {parse_ohlc_field(row['Close'])}")
    print(f"  Volume: {row['Volume']}")
    print(f"  Amount: {row['Amount']}")
