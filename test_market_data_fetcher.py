#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试市场数据获取器
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入配置和数据获取器
from t0t_trading_system.config.config import DATA_CONFIG
from t0t_trading_system.data.fetcher.market_data import MarketDataFetcher, XTP_AVAILABLE

def test_xtp_index_data():
    """测试XTP指数数据获取"""
    if not XTP_AVAILABLE:
        logger.error("XTP API不可用，无法进行测试")
        return
        
    # 创建数据获取器
    fetcher = MarketDataFetcher(DATA_CONFIG, data_source="xtp")
    
    # 获取上证指数数据
    logger.info("正在获取上证指数数据...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    df = fetcher.get_index_data(symbol="000001.SH", freq="D", start_date=start_date, end_date=end_date)
    
    # 打印数据
    if df is not None and not df.empty:
        logger.info(f"获取到{len(df)}条上证指数数据")
        logger.info(f"数据范围: {df.index.min()} - {df.index.max()}")
        logger.info(f"数据示例:\n{df.head()}")
    else:
        logger.warning("未获取到上证指数数据")
    
def test_xtp_stock_data():
    """测试XTP个股数据获取"""
    if not XTP_AVAILABLE:
        logger.error("XTP API不可用，无法进行测试")
        return
        
    # 创建数据获取器
    fetcher = MarketDataFetcher(DATA_CONFIG, data_source="xtp")
    
    # 获取个股数据
    logger.info("正在获取个股数据...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # 测试几只常见股票
    stocks = ["600000.SH", "000001.SZ", "600519.SH"]
    
    for stock in stocks:
        logger.info(f"正在获取{stock}的数据...")
        df = fetcher.get_stock_data(symbol=stock, freq="D", start_date=start_date, end_date=end_date)
        
        # 打印数据
        if df is not None and not df.empty:
            logger.info(f"获取到{len(df)}条{stock}数据")
            logger.info(f"数据范围: {df.index.min()} - {df.index.max()}")
            logger.info(f"数据示例:\n{df.head()}")
        else:
            logger.warning(f"未获取到{stock}数据")
        
        # 等待一段时间，避免请求过于频繁
        time.sleep(2)

def main():
    """主函数"""
    logger.info("开始测试XTP数据源...")
    
    # 测试指数数据获取
    test_xtp_index_data()
    
    # 测试个股数据获取
    test_xtp_stock_data()
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
