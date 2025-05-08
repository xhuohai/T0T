#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
打印XTP配置信息
"""

import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从配置文件中读取配置
try:
    from t0t_trading_system.config.config import DATA_CONFIG
    logger.info("Successfully imported DATA_CONFIG")
except ImportError as e:
    logger.error(f"Error importing DATA_CONFIG: {e}")
    sys.exit(1)

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

if __name__ == "__main__":
    main()
