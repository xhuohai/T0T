#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单测试XTP API导入
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
    import vnxtpquote
    logger.info("Successfully imported vnxtpquote")
except ImportError as e:
    logger.error(f"Error importing vnxtpquote: {e}")
    sys.exit(1)

logger.info("XTP API test completed successfully")
