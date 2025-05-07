"""
市场数据获取模块
支持获取上证指数和个股的历史数据和实时数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("Warning: tushare not installed, using mock data for testing")

# 尝试导入其他可能的数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False


class MarketDataFetcher:
    """市场数据获取类"""

    def __init__(self, config, data_source="tushare", token=None):
        """
        初始化数据获取器

        Args:
            config: 配置信息
            data_source: 数据源，支持 "tushare", "akshare", "baostock", "mock"
            token: API token，如果使用tushare需要提供
        """
        self.config = config
        self.data_source = data_source
        self.cache_dir = config.get("data_dir", "data/storage")
        self.use_cache = config.get("cache_data", True)

        # 确保缓存目录存在
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化数据源
        if data_source == "tushare" and TUSHARE_AVAILABLE:
            if token:
                ts.set_token(token)
            self.pro = ts.pro_api()
        elif data_source == "baostock" and BAOSTOCK_AVAILABLE:
            bs.login()

    def __del__(self):
        """析构函数，释放资源"""
        if self.data_source == "baostock" and BAOSTOCK_AVAILABLE:
            bs.logout()

    def _get_cache_path(self, symbol, freq, start_date, end_date):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{symbol}_{freq}_{start_date}_{end_date}.csv")

    def _load_from_cache(self, symbol, freq, start_date, end_date):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(symbol, freq, start_date, end_date)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, df, symbol, freq, start_date, end_date):
        """保存数据到缓存"""
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, freq, start_date, end_date)
            df.to_csv(cache_path)

    def get_index_data(self, symbol="000001.SH", freq="D", start_date=None, end_date=None, adjust=None):
        """
        获取指数数据

        Args:
            symbol: 指数代码，默认上证指数
            freq: 频率，支持 "D"(日), "W"(周), "M"(月)
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            adjust: 复权方式，None(不复权), "qfq"(前复权), "hfq"(后复权)

        Returns:
            pandas.DataFrame: 指数数据，包含 open, high, low, close, volume 等字段
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

        # 尝试从缓存加载
        if self.use_cache:
            cached_data = self._load_from_cache(symbol, freq, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # 根据数据源获取数据
        if self.data_source == "tushare" and TUSHARE_AVAILABLE:
            return self._get_index_data_tushare(symbol, freq, start_date, end_date)
        elif self.data_source == "akshare" and AKSHARE_AVAILABLE:
            return self._get_index_data_akshare(symbol, freq, start_date, end_date)
        elif self.data_source == "baostock" and BAOSTOCK_AVAILABLE:
            return self._get_index_data_baostock(symbol, freq, start_date, end_date)
        else:
            # 使用模拟数据用于测试
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_index_data_tushare(self, symbol, freq, start_date, end_date):
        """使用tushare获取指数数据"""
        # 转换频率格式
        freq_map = {"D": "daily", "W": "weekly", "M": "monthly"}
        ts_freq = freq_map.get(freq, "daily")

        try:
            if ts_freq == "daily":
                df = self.pro.index_daily(ts_code=symbol, start_date=start_date.replace("-", ""),
                                         end_date=end_date.replace("-", ""))
            else:
                # 对于周线和月线，需要从日线数据转换
                df = self.pro.index_daily(ts_code=symbol, start_date=start_date.replace("-", ""),
                                         end_date=end_date.replace("-", ""))
                if ts_freq == "weekly":
                    df = df.set_index('trade_date')
                    df.index = pd.to_datetime(df.index)
                    df = df.resample('W').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'vol': 'sum'
                    })
                elif ts_freq == "monthly":
                    df = df.set_index('trade_date')
                    df.index = pd.to_datetime(df.index)
                    df = df.resample('M').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'vol': 'sum'
                    })

            # 重命名列
            df = df.rename(columns={
                'vol': 'volume',
                'trade_date': 'date'
            })

            # 确保日期是索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            # 按日期排序
            df = df.sort_index()

            # 缓存数据
            self._save_to_cache(df, symbol, freq, start_date, end_date)

            return df

        except Exception as e:
            print(f"Error fetching data from tushare: {e}")
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_index_data_akshare(self, symbol, freq, start_date, end_date):
        """使用akshare获取指数数据"""
        try:
            import akshare as ak

            # 转换频率格式
            freq_map = {"D": "daily", "W": "weekly", "M": "monthly"}
            ak_freq = freq_map.get(freq, "daily")

            # 转换指数代码格式
            # akshare的指数代码格式与tushare不同，需要转换
            # 例如：000001.SH -> sh000001
            if symbol.endswith('.SH'):
                ak_symbol = f"sh{symbol.split('.')[0]}"
            elif symbol.endswith('.SZ'):
                ak_symbol = f"sz{symbol.split('.')[0]}"
            elif symbol.startswith('sh') or symbol.startswith('sz'):
                ak_symbol = symbol
            else:
                ak_symbol = symbol

            # 尝试不同的指数数据获取方法
            try:
                # 首先尝试使用stock_zh_index_daily获取数据
                if ak_freq == "daily":
                    # 获取日线数据
                    df = ak.stock_zh_index_daily(symbol=ak_symbol)

                    # 转换日期格式
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')

                    # 筛选日期范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                elif ak_freq == "weekly" or ak_freq == "monthly":
                    # 获取日线数据
                    df_daily = ak.stock_zh_index_daily(symbol=ak_symbol)
                    df_daily['date'] = pd.to_datetime(df_daily['date'])
                    df_daily = df_daily.set_index('date')

                    # 筛选日期范围
                    df_daily = df_daily[(df_daily.index >= start_date) & (df_daily.index <= end_date)]

                    # 转换为周线或月线数据
                    resample_rule = 'W' if ak_freq == "weekly" else 'M'
                    df = df_daily.resample(resample_rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
            except Exception as e1:
                print(f"Error using stock_zh_index_daily for {ak_symbol}: {e1}")
                try:
                    # 尝试使用stock_zh_index_daily_em获取数据
                    if ak_symbol.startswith('sh'):
                        em_symbol = ak_symbol.replace('sh', '1')
                    elif ak_symbol.startswith('sz'):
                        em_symbol = ak_symbol.replace('sz', '0')
                    else:
                        em_symbol = ak_symbol

                    # 获取日线数据
                    df = ak.stock_zh_index_daily_em(symbol=em_symbol)

                    # 转换日期格式
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')

                    # 筛选日期范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                    # 如果需要周线或月线，从日线数据转换
                    if ak_freq == "weekly":
                        df = df.resample('W').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        })
                    elif ak_freq == "monthly":
                        df = df.resample('M').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        })
                except Exception as e2:
                    print(f"Error using stock_zh_index_daily_em for {em_symbol}: {e2}")
                    # 尝试使用通用指数接口
                    try:
                        # 尝试使用stock_zh_index_hist_csindex获取数据
                        df = ak.stock_zh_index_hist_csindex(symbol=ak_symbol.upper(), start_date=start_date, end_date=end_date)

                        # 重命名列
                        df = df.rename(columns={
                            '日期': 'date',
                            '开盘': 'open',
                            '最高': 'high',
                            '最低': 'low',
                            '收盘': 'close',
                            '成交量': 'volume'
                        })

                        # 转换日期格式
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')

                        # 如果需要周线或月线，从日线数据转换
                        if ak_freq == "weekly":
                            df = df.resample('W').agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            })
                        elif ak_freq == "monthly":
                            df = df.resample('M').agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            })
                    except Exception as e3:
                        print(f"Error using stock_zh_index_hist_csindex for {ak_symbol}: {e3}")
                        # 所有方法都失败，使用模拟数据
                        return self._get_mock_data(symbol, freq, start_date, end_date)

            # 检查列名是否需要重命名
            if '日期' in df.columns:
                # 重命名列
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })

                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            # 确保数据按日期排序
            df = df.sort_index()

            # 保留需要的列
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

            # 检查是否有缺失的列，如果有则添加
            for col in columns_to_keep:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0  # 成交量缺失时设为0
                    else:
                        # 其他价格列缺失时，使用close列填充
                        df[col] = df['close'] if 'close' in df.columns else 0

            # 缓存数据
            self._save_to_cache(df, symbol, freq, start_date, end_date)

            return df

        except Exception as e:
            print(f"Error fetching data from akshare: {e}")
            # 如果获取失败，使用模拟数据
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_index_data_baostock(self, symbol, freq, start_date, end_date):
        """使用baostock获取指数数据"""
        # 实现baostock数据获取逻辑
        # 由于代码较长，这里仅作为占位符
        return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_mock_data(self, symbol, freq, start_date, end_date):
        """生成模拟数据用于测试"""
        # 生成日期范围
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if freq == "D":
            dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        elif freq == "W":
            dates = pd.date_range(start=start, end=end, freq='W')
        elif freq == "M":
            dates = pd.date_range(start=start, end=end, freq='M')

        # 生成随机价格数据
        n = len(dates)
        np.random.seed(42)  # 固定随机种子以便复现

        # 生成一个随机游走序列作为收盘价
        close = np.random.normal(0, 1, n).cumsum() + 3000

        # 生成其他价格数据
        high = close + np.random.uniform(10, 50, n)
        low = close - np.random.uniform(10, 50, n)
        open_price = low + np.random.uniform(0, 1, n) * (high - low)
        volume = np.random.uniform(1000000, 10000000, n)

        # 创建DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    def get_stock_data(self, symbol, freq="D", start_date=None, end_date=None, adjust="qfq"):
        """
        获取个股数据

        Args:
            symbol: 股票代码
            freq: 频率，支持 "D"(日), "W"(周), "M"(月), "min"(分钟)
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            adjust: 复权方式，None(不复权), "qfq"(前复权), "hfq"(后复权)

        Returns:
            pandas.DataFrame: 股票数据，包含 open, high, low, close, volume 等字段
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

        # 尝试从缓存加载
        if self.use_cache:
            cached_data = self._load_from_cache(symbol, freq, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # 根据数据源获取数据
        if self.data_source == "tushare" and TUSHARE_AVAILABLE:
            return self._get_stock_data_tushare(symbol, freq, start_date, end_date, adjust)
        elif self.data_source == "akshare" and AKSHARE_AVAILABLE:
            return self._get_stock_data_akshare(symbol, freq, start_date, end_date, adjust)
        elif self.data_source == "baostock" and BAOSTOCK_AVAILABLE:
            return self._get_stock_data_baostock(symbol, freq, start_date, end_date, adjust)
        else:
            # 使用模拟数据用于测试
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_stock_data_tushare(self, symbol, freq, start_date, end_date, adjust):
        """使用tushare获取个股数据"""
        # 转换频率格式
        freq_map = {"D": "daily", "W": "weekly", "M": "monthly"}
        ts_freq = freq_map.get(freq, "daily")

        try:
            if ts_freq == "daily":
                if adjust == "qfq":
                    df = ts.pro_bar(ts_code=symbol, adj='qfq', start_date=start_date.replace("-", ""),
                                    end_date=end_date.replace("-", ""))
                elif adjust == "hfq":
                    df = ts.pro_bar(ts_code=symbol, adj='hfq', start_date=start_date.replace("-", ""),
                                    end_date=end_date.replace("-", ""))
                else:
                    df = ts.pro_bar(ts_code=symbol, start_date=start_date.replace("-", ""),
                                    end_date=end_date.replace("-", ""))
            else:
                # 对于周线和月线，需要从日线数据转换
                df = ts.pro_bar(ts_code=symbol, adj=adjust, start_date=start_date.replace("-", ""),
                                end_date=end_date.replace("-", ""))
                if ts_freq == "weekly":
                    df = df.set_index('trade_date')
                    df.index = pd.to_datetime(df.index)
                    df = df.resample('W').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'vol': 'sum'
                    })
                elif ts_freq == "monthly":
                    df = df.set_index('trade_date')
                    df.index = pd.to_datetime(df.index)
                    df = df.resample('M').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'vol': 'sum'
                    })

            # 重命名列
            df = df.rename(columns={
                'vol': 'volume',
                'trade_date': 'date'
            })

            # 确保日期是索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            # 按日期排序
            df = df.sort_index()

            # 缓存数据
            self._save_to_cache(df, symbol, freq, start_date, end_date)

            return df

        except Exception as e:
            print(f"Error fetching data from tushare: {e}")
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def _get_stock_data_akshare(self, symbol, freq, start_date, end_date, adjust):
        """使用akshare获取个股数据"""
        try:
            import akshare as ak

            # 转换频率格式
            freq_map = {"D": "daily", "W": "weekly", "M": "monthly", "min": "minute"}
            ak_freq = freq_map.get(freq, "daily")

            # 转换股票代码格式
            # akshare的股票代码格式为：sh600000或sz000001
            if symbol.endswith('.SH'):
                ak_symbol = f"sh{symbol.split('.')[0]}"
            elif symbol.endswith('.SZ'):
                ak_symbol = f"sz{symbol.split('.')[0]}"
            else:
                # 如果没有后缀，根据第一位判断
                if symbol.startswith('6'):
                    ak_symbol = f"sh{symbol}"
                elif symbol.startswith('sh') or symbol.startswith('sz'):
                    ak_symbol = symbol
                else:
                    ak_symbol = f"sz{symbol}"

            # 获取数据
            try:
                if ak_freq == "minute":
                    # 获取分钟级数据
                    # 注意：akshare的分钟数据接口只能获取最近几天的数据
                    try:
                        # 尝试使用东方财富网的分钟数据接口
                        print(f"Fetching minute data for {ak_symbol}")

                        # 直接使用最简单的接口获取最近的分钟数据
                        # 不指定日期范围，让akshare自动获取最近的数据
                        df = ak.stock_zh_a_hist_min_em(symbol=ak_symbol, period='1')

                        # 重命名列
                        if '日期' in df.columns:
                            df = df.rename(columns={
                                '日期': 'datetime',
                                '开盘': 'open',
                                '最高': 'high',
                                '最低': 'low',
                                '收盘': 'close',
                                '成交量': 'volume'
                            })

                        # 转换日期格式
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.set_index('datetime')

                        print(f"Successfully fetched {len(df)} minute data points")

                    except Exception as e:
                        print(f"Error fetching minute data from akshare: {e}")
                        print("Falling back to mock minute data...")
                        # 如果获取分钟数据失败，使用日线数据生成模拟分钟数据
                        daily_data = self._get_stock_data_akshare(symbol, "D", start_date, end_date, adjust)
                        from t0t_trading_system.main import generate_mock_minute_data
                        df = generate_mock_minute_data(daily_data)

                elif ak_freq == "daily":
                    # 获取日线数据
                    if adjust == "qfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                    elif adjust == "hfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
                    else:
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
                elif ak_freq == "weekly":
                    # 获取周线数据
                    if adjust == "qfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="weekly", start_date=start_date, end_date=end_date, adjust="qfq")
                    elif adjust == "hfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="weekly", start_date=start_date, end_date=end_date, adjust="hfq")
                    else:
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="weekly", start_date=start_date, end_date=end_date, adjust="")
                elif ak_freq == "monthly":
                    # 获取月线数据
                    if adjust == "qfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="monthly", start_date=start_date, end_date=end_date, adjust="qfq")
                    elif adjust == "hfq":
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="monthly", start_date=start_date, end_date=end_date, adjust="hfq")
                    else:
                        df = ak.stock_zh_a_hist(symbol=ak_symbol, period="monthly", start_date=start_date, end_date=end_date, adjust="")
            except Exception as e:
                if ak_freq != "minute":  # 已经在上面处理了分钟数据的异常
                    print(f"Error fetching {ak_freq} data for {ak_symbol}: {e}")
                    # 尝试使用通用的历史数据接口
                    if adjust == "qfq":
                        df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="qfq")
                    elif adjust == "hfq":
                        df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="hfq")
                    else:
                        df = ak.stock_zh_a_daily(symbol=ak_symbol, adjust="")

                    # 筛选日期范围
                    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

                    # 如果需要周线或月线，从日线数据转换
                    if ak_freq == "weekly":
                        df = df.resample('W').agg({
                            'open': 'first', 'high': 'max', 'low': 'min',
                            'close': 'last', 'volume': 'sum'
                        })
                    elif ak_freq == "monthly":
                        df = df.resample('M').agg({
                            'open': 'first', 'high': 'max', 'low': 'min',
                            'close': 'last', 'volume': 'sum'
                        })

            # 检查列名是否为中文，如果是则重命名
            if '日期' in df.columns:
                # 重命名列
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })

                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            # 确保数据按日期排序
            df = df.sort_index()

            # 保留需要的列
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

            # 检查是否有缺失的列，如果有则添加
            for col in columns_to_keep:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0  # 成交量缺失时设为0
                    else:
                        # 其他价格列缺失时，使用close列填充
                        df[col] = df['close'] if 'close' in df.columns else 0

            # 缓存数据
            self._save_to_cache(df, symbol, freq, start_date, end_date)

            return df

        except Exception as e:
            print(f"Error fetching data from akshare: {e}")
            # 如果获取失败，使用模拟数据
            return self._get_mock_data(symbol, freq, start_date, end_date)

    def get_realtime_data(self, symbols):
        """
        获取实时行情数据

        Args:
            symbols: 股票代码列表

        Returns:
            pandas.DataFrame: 实时行情数据
        """
        if self.data_source == "akshare" and AKSHARE_AVAILABLE:
            try:
                import akshare as ak

                # 转换股票代码格式
                ak_symbols = []
                for symbol in symbols:
                    if symbol.endswith('.SH'):
                        ak_symbols.append(f"sh{symbol.split('.')[0]}")
                    elif symbol.endswith('.SZ'):
                        ak_symbols.append(f"sz{symbol.split('.')[0]}")
                    else:
                        # 如果没有后缀，根据第一位判断
                        if symbol.startswith('6'):
                            ak_symbols.append(f"sh{symbol}")
                        else:
                            ak_symbols.append(f"sz{symbol}")

                # 获取实时行情数据
                df = ak.stock_zh_a_spot_em()

                # 筛选需要的股票
                df = df[df['代码'].isin(ak_symbols)]

                # 重命名列
                df = df.rename(columns={
                    '代码': 'code',
                    '名称': 'name',
                    '最新价': 'price',
                    '涨跌幅': 'change',
                    '成交量': 'volume',
                    '时间': 'time'
                })

                # 设置索引
                df = df.set_index('code')

                # 转换数据类型
                df['price'] = df['price'].astype(float)
                df['change'] = df['change'].astype(float)
                df['volume'] = df['volume'].astype(float)

                # 添加时间戳
                if 'time' not in df.columns:
                    df['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                return df

            except Exception as e:
                print(f"Error fetching realtime data from akshare: {e}")
                # 如果获取失败，使用模拟数据
                return self._get_mock_realtime_data(symbols)
        else:
            # 使用模拟数据
            return self._get_mock_realtime_data(symbols)

    def _get_mock_realtime_data(self, symbols):
        """生成模拟实时数据"""
        data = {}
        for symbol in symbols:
            data[symbol] = {
                'code': symbol,
                'name': f'Stock {symbol}',
                'price': round(np.random.uniform(10, 100), 2),
                'change': round(np.random.uniform(-5, 5), 2),
                'volume': int(np.random.uniform(1000000, 10000000)),
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        return pd.DataFrame.from_dict(data, orient='index')
