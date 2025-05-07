import akshare as ak
import pandas as pd

def test_minute_data():
    """测试获取分钟数据"""
    try:
        # 尝试获取贵州茅台的分钟数据
        print("尝试获取贵州茅台(sh600519)的分钟数据...")
        df = ak.stock_zh_a_hist_min_em(symbol="sh600519", period='1')
        
        # 打印数据信息
        print(f"获取到 {len(df)} 条分钟数据")
        print("数据列名:", df.columns.tolist())
        print("数据前5行:")
        print(df.head())
        
        return True
    except Exception as e:
        print(f"获取分钟数据失败: {e}")
        return False

def test_daily_data():
    """测试获取日线数据"""
    try:
        # 尝试获取贵州茅台的日线数据
        print("\n尝试获取贵州茅台(sh600519)的日线数据...")
        df = ak.stock_zh_a_hist(symbol="sh600519", period="daily", start_date="2024-04-01", end_date="2024-05-07", adjust="qfq")
        
        # 打印数据信息
        print(f"获取到 {len(df)} 条日线数据")
        print("数据列名:", df.columns.tolist())
        print("数据前5行:")
        print(df.head())
        
        return True
    except Exception as e:
        print(f"获取日线数据失败: {e}")
        return False

if __name__ == "__main__":
    print("测试akshare数据获取功能")
    
    # 测试分钟数据
    minute_success = test_minute_data()
    
    # 测试日线数据
    daily_success = test_daily_data()
    
    # 打印测试结果
    print("\n测试结果:")
    print(f"分钟数据测试: {'成功' if minute_success else '失败'}")
    print(f"日线数据测试: {'成功' if daily_success else '失败'}")
