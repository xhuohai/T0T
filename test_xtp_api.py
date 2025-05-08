#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试XTP API连接和数据获取
"""

import os
import sys
import time
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入XTP API
try:
    from vnxtpquote import *
    logger.info("Successfully imported vnxtpquote")
except ImportError as e:
    logger.error(f"Error importing vnxtpquote: {e}")
    sys.exit(1)

# 从配置文件中读取配置
from t0t_trading_system.config.config import DATA_CONFIG

class XTPQuoteDemo(QuoteApi):
    """XTP行情API演示类"""

    def __init__(self):
        """初始化"""
        super(XTPQuoteDemo, self).__init__()
        self.connected = False
        self.subscribed_symbols = set()
        self.market_data = {}

    def onDisconnected(self, reason):
        """当客户端与行情后台通信连接断开时，该方法被调用"""
        logger.warning(f"XTP行情服务断开连接，原因: {reason}")
        self.connected = False

    def onError(self, data):
        """错误应答"""
        if data and 'error_id' in data and data['error_id'] != 0:
            logger.error(f"XTP行情API错误: {data['error_id']} - {data['error_msg']}")

    def onSubMarketData(self, data, error, last):
        """订阅行情应答"""
        if error and 'error_id' in error and error['error_id'] != 0:
            logger.error(f"订阅行情失败: {error['error_id']} - {error['error_msg']}")
        else:
            symbol = data['ticker']
            exchange = data['exchange_id']
            logger.info(f"成功订阅行情: {symbol} - {exchange}")
            self.subscribed_symbols.add(symbol)

    def onDepthMarketData(self, data, bid1_qty_list, bid1_counts, max_bid1_count, ask1_qty_list, ask1_count, max_ask1_count):
        """深度行情通知"""
        if data:
            symbol = data['ticker']
            logger.info(f"收到行情数据: {symbol} - 最新价: {data['last_price']}")

            # 存储行情数据
            if symbol not in self.market_data:
                self.market_data[symbol] = []

            # 转换为标准格式
            market_data = {
                'datetime': datetime.strptime(data['data_time'], '%Y%m%d%H%M%S%f') if 'data_time' in data else datetime.now(),
                'open': data['open_price'],
                'high': data['high_price'],
                'low': data['low_price'],
                'close': data['last_price'],
                'volume': data['qty'],
                'amount': data['turnover'],
                'bid': data['bid'][0],
                'ask': data['ask'][0],
                'bid_vol': data['bid_qty'][0],
                'ask_vol': data['ask_qty'][0],
            }

            self.market_data[symbol].append(market_data)

    def onQueryAllTickers(self, data, error, last):
        """查询可交易合约的应答"""
        if error and 'error_id' in error and error['error_id'] != 0:
            logger.error(f"查询合约失败: {error['error_id']} - {error['error_msg']}")
        else:
            if data:
                logger.info(f"合约信息: {data['ticker']} - {data['ticker_name']}")

def main():
    """主函数"""
    # 获取XTP配置
    xtp_config = DATA_CONFIG.get("xtp", {})

    # 创建API实例
    api = XTPQuoteDemo()
    api.createQuoteApi(1, os.path.join(os.getcwd(), "xtp_log"), 4)

    # 设置心跳间隔
    api.setHeartBeatInterval(15)

    # 设置UDP缓冲区大小
    api.setUDPBufferSize(128)

    # 登录
    ip = xtp_config.get("quote_ip", "")
    port = xtp_config.get("quote_port", 0)
    user = xtp_config.get("user", "")
    password = xtp_config.get("password", "")
    local_ip = xtp_config.get("local_ip", "")

    # 打印配置信息
    logger.info(f"XTP配置信息:")
    logger.info(f"  行情服务器IP: {ip}")
    logger.info(f"  行情服务器端口: {port}")
    logger.info(f"  用户名: {user}")
    logger.info(f"  本地IP: {local_ip}")

    logger.info(f"正在连接XTP行情服务器: {ip}:{port}")
    ret = api.login(ip, port, user, password, 1, local_ip)

    if ret != 0:
        error = api.getApiLastError()
        logger.error(f"XTP行情服务登录失败: {error['error_id']} - {error['error_msg']}")
        return

    logger.info("XTP行情服务登录成功")
    api.connected = True

    # 查询上交所股票列表
    logger.info("正在查询上交所股票列表...")
    ret = api.queryAllTickers(1)  # 1-上交所
    if ret != 0:
        error = api.getApiLastError()
        logger.error(f"查询上交所股票列表失败: {error['error_id']} - {error['error_msg']}")

    # 等待数据返回
    time.sleep(1)

    # 查询深交所股票列表
    logger.info("正在查询深交所股票列表...")
    ret = api.queryAllTickers(2)  # 2-深交所
    if ret != 0:
        error = api.getApiLastError()
        logger.error(f"查询深交所股票列表失败: {error['error_id']} - {error['error_msg']}")

    # 等待数据返回
    time.sleep(1)

    # 订阅上证指数行情
    logger.info("正在订阅上证指数行情...")
    ret = api.subscribeMarketData(["000001"], 1, 1)  # 上证指数
    if ret != 0:
        error = api.getApiLastError()
        logger.error(f"订阅上证指数行情失败: {error['error_id']} - {error['error_msg']}")

    # 等待行情数据
    logger.info("等待行情数据...")
    time.sleep(10)

    # 登出
    api.logout()
    logger.info("XTP行情服务登出成功")

if __name__ == "__main__":
    main()
