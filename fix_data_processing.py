#!/usr/bin/env python3
"""
修复数据处理脚本 - 正确解析extracted原始数据
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import re

def parse_extracted_csv(file_path):
    """
    解析extracted目录中格式有问题的CSV文件

    原始数据格式分析：
    Time,Open,High,Low,Close,Volume,Amount,TVolume,TAmount,
    2023/1/3 9:31:00,3087.5103087.510,3078.7703083.870,1002458.00010024588288.000,1002458.00010024588288.000,

    问题：
    - 第2列：Open+High合并 (3087.5103087.510)
    - 第3列：Low+Close合并 (3078.7703083.870)
    - 第4列：Volume+Amount合并 (1002458.00010024588288.000)
    - 第5列：TVolume+TAmount合并 (1002458.00010024588288.000)
    """
    print(f"正在处理文件: {file_path}")

    try:
        # 读取原始文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 解析数据
        parsed_data = []

        for i, line in enumerate(lines):
            if i == 0:  # 跳过标题行
                continue

            line = line.strip()
            if not line or line.endswith(','):
                line = line.rstrip(',')

            parts = line.split(',')
            if len(parts) < 4:
                continue

            try:
                # 解析时间
                time_str = parts[0]

                # 解析Open+High (第2列)
                open_high_str = parts[1]
                # 格式: 3087.5103087.510 -> 3087.510 和 3087.510
                if '.' in open_high_str:
                    # 找到所有数字和小数点的位置
                    digits = re.findall(r'\d+\.\d+', open_high_str)
                    if len(digits) >= 2:
                        open_price = float(digits[0])
                        high_price = float(digits[1])
                    elif len(digits) == 1:
                        # 如果只找到一个数字，可能是格式问题，尝试手动分割
                        num_str = digits[0]
                        if len(open_high_str) > len(num_str):
                            # 尝试从中间分割
                            mid = len(open_high_str) // 2
                            try:
                                open_price = float(open_high_str[:mid])
                                high_price = float(open_high_str[mid:])
                            except:
                                open_price = high_price = float(num_str)
                        else:
                            open_price = high_price = float(num_str)
                    else:
                        continue
                else:
                    continue

                # 解析Low+Close (第3列)
                low_close_str = parts[2]
                digits = re.findall(r'\d+\.\d+', low_close_str)
                if len(digits) >= 2:
                    low_price = float(digits[0])
                    close_price = float(digits[1])
                elif len(digits) == 1:
                    num_str = digits[0]
                    if len(low_close_str) > len(num_str):
                        mid = len(low_close_str) // 2
                        try:
                            low_price = float(low_close_str[:mid])
                            close_price = float(low_close_str[mid:])
                        except:
                            low_price = close_price = float(num_str)
                    else:
                        low_price = close_price = float(num_str)
                else:
                    continue

                # 解析Volume+Amount (第4列)
                vol_amt_str = parts[3]
                digits = re.findall(r'\d+\.\d+', vol_amt_str)
                if len(digits) >= 2:
                    volume = float(digits[0])
                    amount = float(digits[1])
                elif len(digits) == 1:
                    # 对于Volume+Amount，通常Volume较小，Amount较大
                    # 尝试智能分割
                    full_str = vol_amt_str
                    if '000' in full_str:
                        # 寻找连续的000作为分割点
                        parts_split = full_str.split('000')
                        if len(parts_split) >= 2:
                            try:
                                volume = float(parts_split[0] + '000')
                                amount = float(''.join(parts_split[1:]))
                            except:
                                volume = amount = float(digits[0])
                        else:
                            volume = amount = float(digits[0])
                    else:
                        volume = amount = float(digits[0])
                else:
                    volume = amount = 0.0

                # 解析TVolume+TAmount (第5列，如果存在)
                if len(parts) > 4:
                    tvol_tamt_str = parts[4]
                    digits = re.findall(r'\d+\.\d+', tvol_tamt_str)
                    if len(digits) >= 2:
                        tvolume = float(digits[0])
                        tamount = float(digits[1])
                    elif len(digits) == 1:
                        if '000' in tvol_tamt_str:
                            parts_split = tvol_tamt_str.split('000')
                            if len(parts_split) >= 2:
                                try:
                                    tvolume = float(parts_split[0] + '000')
                                    tamount = float(''.join(parts_split[1:]))
                                except:
                                    tvolume = tamount = float(digits[0])
                            else:
                                tvolume = tamount = float(digits[0])
                        else:
                            tvolume = tamount = float(digits[0])
                    else:
                        tvolume = tamount = 0.0
                else:
                    tvolume = volume
                    tamount = amount

                # 添加解析后的数据
                parsed_data.append([
                    time_str, open_price, high_price, low_price, close_price,
                    volume, amount, tvolume, tamount
                ])

            except (ValueError, IndexError) as e:
                print(f"解析第{i+1}行时出错: {line[:50]}... 错误: {e}")
                continue

        if not parsed_data:
            print("没有成功解析任何数据")
            return None

        # 创建DataFrame
        header = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TVolume', 'TAmount']
        df = pd.DataFrame(parsed_data, columns=header)

        # 转换时间格式
        df['datetime'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
        df = df.drop('Time', axis=1)

        # 重新排列列
        df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TVolume', 'TAmount']]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'tvolume', 'tamount']

        print(f"成功解析 {len(df)} 行数据")
        return df

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_multiple_symbols(symbols):
    """处理多个股票代码的数据"""
    extracted_base = "/home/chenghai/Work/LLM/data/extracted"
    output_dir = "data/fixed_processed"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"开始处理 {symbol}")
        print(f"{'='*60}")

        all_data = []

        for year in ['2023', '2024']:
            year_path = os.path.join(extracted_base, year)
            if not os.path.exists(year_path):
                continue

            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue

                for day in os.listdir(month_path):
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path):
                        continue

                    symbol_file = os.path.join(day_path, f"{symbol}.csv")
                    if os.path.exists(symbol_file):
                        df = parse_extracted_csv(symbol_file)
                        if df is not None and len(df) > 0:
                            all_data.append(df)

        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('datetime')

            # 保存到文件
            output_file = os.path.join(output_dir, f"{symbol}.csv")
            combined_df.to_csv(output_file, index=False)

            print(f"✅ {symbol} 处理完成:")
            print(f"   总共处理了 {len(combined_df)} 行数据")
            print(f"   数据时间范围: {combined_df['datetime'].min()} 到 {combined_df['datetime'].max()}")
            print(f"   每日平均数据点: {len(combined_df) / combined_df['datetime'].dt.date.nunique():.1f}")
            print(f"   保存到: {output_file}")

            results[symbol] = {
                'rows': len(combined_df),
                'start_date': combined_df['datetime'].min(),
                'end_date': combined_df['datetime'].max(),
                'file_path': output_file
            }
        else:
            print(f"❌ {symbol} 没有找到有效数据")
            results[symbol] = None

    return results

def process_all_extracted_data():
    """处理所有extracted数据 - 保持向后兼容"""
    return process_multiple_symbols(['SH000001'])

def test_single_file():
    """测试单个文件的解析"""
    test_file = "/home/chenghai/Work/LLM/data/extracted/2023/01/2023-01-03/SH000001.csv"
    print(f"测试解析单个文件: {test_file}")

    df = parse_extracted_csv(test_file)
    if df is not None:
        print("\n解析结果:")
        print(df.head(10))
        print(f"\n数据统计:")
        print(f"总行数: {len(df)}")
        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        return True
    return False

if __name__ == "__main__":
    print("开始修复数据处理...")

    # 定义要处理的10支股票
    target_symbols = [
        'SH600036',  # 招商银行 (主要回测标的)
        'SH600000',  # 浦发银行
        'SH600519',  # 贵州茅台
        'SH600030',  # 中信证券
        'SH600887',  # 伊利股份
        'SH600276',  # 恒瑞医药
        'SH600585',  # 海螺水泥
        'SH600104',  # 上汽集团
        'SH600050',  # 中国联通
        'SH000001'   # 上证指数 (基准)
    ]

    # 先测试单个文件
    if test_single_file():
        print("\n单文件测试成功，开始处理10支股票数据...")
        results = process_multiple_symbols(target_symbols)

        print(f"\n{'='*80}")
        print("📊 数据处理汇总报告")
        print(f"{'='*80}")

        successful_count = 0
        total_rows = 0

        for symbol, result in results.items():
            if result:
                successful_count += 1
                total_rows += result['rows']
                print(f"✅ {symbol}: {result['rows']:,} 行数据")
            else:
                print(f"❌ {symbol}: 处理失败")

        print(f"\n📈 处理结果:")
        print(f"   成功处理: {successful_count}/{len(target_symbols)} 支股票")
        print(f"   总数据量: {total_rows:,} 行")
        print(f"   平均每支股票: {total_rows/successful_count:,.0f} 行" if successful_count > 0 else "")

        if successful_count > 0:
            print(f"\n🎯 接下来可以用 SH600036 (招商银行) 进行T0策略回测")

    else:
        print("单文件测试失败，请检查解析逻辑")
