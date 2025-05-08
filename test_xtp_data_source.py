#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试XTP数据源
"""

import os
import sys
import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从配置文件中读取配置
from t0t_trading_system.config.config import DATA_CONFIG

# 导入XTP数据源
from t0t_trading_system.data.fetcher.xtp_data import XTPDataSource, XTP_AVAILABLE

def main():
    """主函数"""
    if not XTP_AVAILABLE:
        logger.error("XTP API不可用，无法进行测试")
        return
        
    # 创建XTP数据源
    xtp = XTPDataSource(DATA_CONFIG)
    
    # 连接XTP行情服务器
    logger.info("正在连接XTP行情服务器...")
    if not xtp.connect():
        logger.error("连接XTP行情服务器失败")
        return
        
    logger.info("连接XTP行情服务器成功")
    
    # 获取上交所股票列表
    logger.info("正在获取上交所股票列表...")
    sh_stocks = xtp.get_stock_list(1)  # 1-上交所
    logger.info(f"获取到{len(sh_stocks)}个上交所股票")
    
    # 打印前10个股票信息
    for i, stock in enumerate(sh_stocks[:10]):
        logger.info(f"股票{i+1}: {stock['code']} - {stock['name']}")
    
    # 获取深交所股票列表
    logger.info("正在获取深交所股票列表...")
    sz_stocks = xtp.get_stock_list(2)  # 2-深交所
    logger.info(f"获取到{len(sz_stocks)}个深交所股票")
    
    # 打印前10个股票信息
    for i, stock in enumerate(sz_stocks[:10]):
        logger.info(f"股票{i+1}: {stock['code']} - {stock['name']}")
    
    # 断开连接
    xtp.disconnect()
    logger.info("断开XTP行情服务器连接")

if __name__ == "__main__":
    main()
