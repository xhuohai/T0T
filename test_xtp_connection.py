#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试XTP API连接
"""

import os
import sys
import time
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

def main():
    """主函数"""
    # 获取XTP配置
    xtp_config = DATA_CONFIG.get("xtp", {})
    
    # 打印配置信息
    logger.info(f"XTP配置信息:")
    logger.info(f"  行情服务器IP: {xtp_config.get('quote_ip', '')}")
    logger.info(f"  行情服务器端口: {xtp_config.get('quote_port', 0)}")
    logger.info(f"  用户名: {xtp_config.get('user', '')}")
    logger.info(f"  本地IP: {xtp_config.get('local_ip', '')}")
    
    # 创建API实例
    api = QuoteApi()
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
    
    logger.info(f"正在连接XTP行情服务器: {ip}:{port}")
    ret = api.login(ip, port, user, password, 1, local_ip)
    
    if ret != 0:
        error = api.getApiLastError()
        logger.error(f"XTP行情服务登录失败: {error['error_id']} - {error['error_msg']}")
        return
        
    logger.info("XTP行情服务登录成功")
    
    # 等待一段时间
    logger.info("等待5秒...")
    time.sleep(5)
    
    # 登出
    api.logout()
    logger.info("XTP行情服务登出成功")

if __name__ == "__main__":
    main()
