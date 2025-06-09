"""
XTP市场数据获取模块
支持通过中泰证券XTP API获取A股市场数据
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入XTP API
try:
    from vnxtpquote import *
    XTP_AVAILABLE = True
    logger.info("Successfully imported vnxtpquote")
except ImportError as e:
    XTP_AVAILABLE = False
    logger.warning(f"Warning: XTP API not installed, using mock data for testing. Error: {e}")
    # 创建一个虚拟的QuoteApi类
    class QuoteApi:
        def __init__(self):
            pass
        def createQuoteApi(self, *args):
            pass
        def setHeartBeatInterval(self, *args):
            pass
        def setUDPBufferSize(self, *args):
            pass
        def login(self, *args):
            return -1
        def logout(self):
            pass
        def getApiLastError(self):
            return {'error_id': -1, 'error_msg': 'XTP API not available'}
        def queryAllTickers(self, *args):
            return -1
        def subscribeMarketData(self, *args):
            return -1

class XTPQuoteApi(QuoteApi):
    """XTP行情API封装类"""

    def __init__(self, config):
        """
        初始化XTP行情API

        Args:
            config: XTP配置信息
        """
        super(XTPQuoteApi, self).__init__()
        self.config = config
        self.connected = False
        self.subscribed_symbols = set()
        self.market_data = {}
        self.all_tickers = []

    def onDisconnected(self, reason):
        """
        当客户端与行情后台通信连接断开时，该方法被调用

        Args:
            reason: 错误原因
        """
        logger.warning(f"XTP行情服务断开连接，原因: {reason}")
        self.connected = False

    def onError(self, data):
        """
        错误应答

        Args:
            data: 错误信息
        """
        if data and 'error_id' in data and data['error_id'] != 0:
            logger.error(f"XTP行情API错误: {data['error_id']} - {data['error_msg']}")

    def onSubMarketData(self, data, error, last):
        """
        订阅行情应答

        Args:
            data: 详细的合约订阅情况
            error: 订阅合约发生错误时的错误信息
            last: 是否此次订阅的最后一个应答
        """
        if error and 'error_id' in error and error['error_id'] != 0:
            logger.error(f"订阅行情失败: {error['error_id']} - {error['error_msg']}")
        else:
            symbol = data['ticker']
            exchange = data['exchange_id']
            logger.info(f"成功订阅行情: {symbol} - {exchange}")
            self.subscribed_symbols.add(symbol)

    def onDepthMarketData(self, data, bid1_qty_list, bid1_counts, max_bid1_count, ask1_qty_list, ask1_count, max_ask1_count):
        """
        深度行情通知

        Args:
            data: 行情数据
            bid1_qty_list: 买一队列数据
            bid1_counts: 买一队列的有效委托笔数
            max_bid1_count: 买一队列总委托笔数
            ask1_qty_list: 卖一队列数据
            ask1_count: 卖一队列的有效委托笔数
            max_ask1_count: 卖一队列总委托笔数
        """
        if data:
            symbol = data['ticker']
            # 转换为标准格式
            market_data = {
                'datetime': pd.to_datetime(data['data_time'], format='%Y%m%d%H%M%S%f'),
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

            # 存储行情数据
            if symbol not in self.market_data:
                self.market_data[symbol] = []
            self.market_data[symbol].append(market_data)

    def onQueryAllTickers(self, data, error, last):
        """
        查询可交易合约的应答

        Args:
            data: 可交易合约信息
            error: 查询可交易合约时发生错误时返回的错误信息
            last: 是否此次查询可交易合约的最后一个应答
        """
        if error and 'error_id' in error and error['error_id'] != 0:
            logger.error(f"查询合约失败: {error['error_id']} - {error['error_msg']}")
        else:
            if data:
                self.all_tickers.append({
                    'exchange_id': data['exchange_id'],
                    'ticker': data['ticker'],
                    'ticker_name': data['ticker_name'],
                    'ticker_type': data['ticker_type'],
                    'pre_close_price': data['pre_close_price'],
                    'upper_limit_price': data['upper_limit_price'],
                    'lower_limit_price': data['lower_limit_price'],
                })

    def onQueryTickersPriceInfo(self, data, error, last):
        """
        查询合约的最新价格信息应答

        Args:
            data: 合约价格信息
            error: 查询合约价格信息时发生错误时返回的错误信息
            last: 是否此次查询的最后一个应答
        """
        if error and 'error_id' in error and error['error_id'] != 0:
            logger.error(f"查询价格信息失败: {error['error_id']} - {error['error_msg']}")
        else:
            if data:
                symbol = data['ticker']
                if symbol not in self.market_data:
                    self.market_data[symbol] = []
                self.market_data[symbol].append({
                    'datetime': datetime.now(),
                    'close': data['last_price'],
                })

class XTPDataSource:
    """XTP数据源类"""

    def __init__(self, config):
        """
        初始化XTP数据源

        Args:
            config: 配置信息
        """
        self.config = config
        self.xtp_config = config.get("xtp", {})
        self.api = None
        self.session_id = None
        self.connected = False

        # 初始化XTP API
        if XTP_AVAILABLE:
            self.init_api()

    def init_api(self):
        """初始化XTP API"""
        try:
            self.api = XTPQuoteApi(self.xtp_config)
            # 创建API实例
            self.api.createQuoteApi(1, os.path.join(os.getcwd(), "xtp_log"), 4)

            # 设置心跳间隔
            self.api.setHeartBeatInterval(15)

            # 设置UDP缓冲区大小
            self.api.setUDPBufferSize(128)

            logger.info("XTP API初始化成功")
        except Exception as e:
            logger.error(f"XTP API初始化失败: {e}")
            self.api = None

    def connect(self):
        """连接XTP行情服务器"""
        if not XTP_AVAILABLE or not self.api:
            logger.warning("XTP API不可用，无法连接")
            return False

        try:
            # 获取配置信息
            ip = self.xtp_config.get("quote_ip", "")
            port = self.xtp_config.get("quote_port", 0)
            user = self.xtp_config.get("user", "")
            password = self.xtp_config.get("password", "")
            local_ip = self.xtp_config.get("local_ip", "")

            # 登录
            ret = self.api.login(ip, port, user, password, 1, local_ip)
            if ret != 0:
                error = self.api.getApiLastError()
                logger.error(f"XTP行情服务登录失败: {error['error_id']} - {error['error_msg']}")
                return False

            logger.info("XTP行情服务登录成功")
            self.connected = True
            self.api.connected = True
            return True
        except Exception as e:
            logger.error(f"XTP行情服务连接异常: {e}")
            return False

    def disconnect(self):
        """断开XTP行情服务器连接"""
        if self.api and self.connected:
            try:
                self.api.logout()
                logger.info("XTP行情服务登出成功")
            except Exception as e:
                logger.error(f"XTP行情服务登出异常: {e}")
            finally:
                self.connected = False
                self.api.connected = False

    def get_stock_list(self, exchange_id=1):
        """
        获取股票列表

        Args:
            exchange_id: 交易所ID，1-上交所，2-深交所

        Returns:
            list: 股票列表
        """
        if not self.connected:
            if not self.connect():
                return []

        try:
            # 清空之前的数据
            self.api.all_tickers = []

            # 查询合约
            logger.info(f"正在查询交易所ID={exchange_id}的股票列表...")
            ret = self.api.queryAllTickers(exchange_id)
            if ret != 0:
                error = self.api.getApiLastError()
                logger.error(f"查询合约失败: {error['error_id']} - {error['error_msg']}")
                return []

            # 等待数据返回
            time.sleep(1)

            logger.info(f"获取到{len(self.api.all_tickers)}个合约信息")

            # 转换为标准格式
            stock_list = []
            for ticker_info in self.api.all_tickers:
                stock = {
                    'code': ticker_info['ticker'],
                    'name': ticker_info['ticker_name'],
                    'exchange': 'SH' if exchange_id == 1 else 'SZ',
                    'pre_close': ticker_info['pre_close_price'],
                    'upper_limit': ticker_info['upper_limit_price'],
                    'lower_limit': ticker_info['lower_limit_price'],
                }
                stock_list.append(stock)

            return stock_list
        except Exception as e:
            logger.error(f"获取股票列表异常: {e}")
            return []
