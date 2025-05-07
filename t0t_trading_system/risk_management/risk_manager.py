"""
风险管理模块
实现错误交易修正和止损逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RiskManager:
    """风险管理类"""
    
    def __init__(self, config):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)
        self.max_drawdown = config.get("max_drawdown", 0.1)
        
        # 风险事件记录
        self.risk_events = []
        
        # 当前持仓风险状态
        self.position_risk = {
            'max_value': 0,
            'current_value': 0,
            'drawdown': 0,
            'stop_loss_triggered': False
        }
    
    def check_trade_failure(self, minute_data, trade_record, higher_tf_data=None):
        """
        检查交易是否失败
        
        Args:
            minute_data: DataFrame，分钟级数据，包含交易信号
            trade_record: dict，交易记录
            higher_tf_data: DataFrame，更高时间框架的数据，如5分钟数据
            
        Returns:
            bool: 交易是否失败
            str: 失败原因
        """
        trade_time = trade_record['time']
        trade_type = trade_record['type']
        trade_price = trade_record['price']
        
        # 获取交易后的数据
        post_trade_data = minute_data.loc[minute_data.index > trade_time]
        
        if post_trade_data.empty:
            return False, "No data after trade"
        
        # 条件A：价格出现相反方向的运动
        if trade_type == 'buy':
            # 买入后价格下跌
            price_moved_opposite = post_trade_data['low'].min() < trade_price
            price_level = post_trade_data['low'].min()
        else:
            # 卖出后价格上涨
            price_moved_opposite = post_trade_data['high'].max() > trade_price
            price_level = post_trade_data['high'].max()
        
        # 条件B：依据的交易周期背离信号消失
        if trade_type == 'buy':
            # 买入信号是底背离，检查底背离是否消失
            divergence_disappeared = not post_trade_data['bullish_divergence'].any()
        else:
            # 卖出信号是顶背离，检查顶背离是否消失
            divergence_disappeared = not post_trade_data['bearish_divergence'].any()
        
        # 条件C：上一层周期的指标未出现背离
        higher_tf_divergence = False
        
        if higher_tf_data is not None:
            # 获取交易时间对应的更高时间框架数据
            higher_tf_bar = higher_tf_data.loc[higher_tf_data.index <= trade_time].iloc[-1]
            
            if trade_type == 'buy':
                higher_tf_divergence = higher_tf_bar.get('bullish_divergence', False)
            else:
                higher_tf_divergence = higher_tf_bar.get('bearish_divergence', False)
        
        # 判断交易是否失败
        is_failed = price_moved_opposite and divergence_disappeared and not higher_tf_divergence
        
        failure_reason = ""
        if is_failed:
            failure_reason = "Trade failed: "
            if price_moved_opposite:
                failure_reason += "Price moved in opposite direction. "
            if divergence_disappeared:
                failure_reason += "Divergence signal disappeared. "
            if not higher_tf_divergence:
                failure_reason += "No higher timeframe divergence confirmation."
        
        return is_failed, failure_reason
    
    def calculate_stop_loss_price(self, trade_record):
        """
        计算止损价格
        
        Args:
            trade_record: dict，交易记录
            
        Returns:
            float: 止损价格
        """
        trade_type = trade_record['type']
        trade_price = trade_record['price']
        
        if trade_type == 'buy':
            # 买入交易的止损价格
            stop_loss_price = trade_price * (1 - self.stop_loss_pct)
        else:
            # 卖出交易的止损价格
            stop_loss_price = trade_price * (1 + self.stop_loss_pct)
        
        return stop_loss_price
    
    def should_stop_loss(self, trade_record, current_price):
        """
        判断是否应该止损
        
        Args:
            trade_record: dict，交易记录
            current_price: float，当前价格
            
        Returns:
            bool: 是否应该止损
        """
        trade_type = trade_record['type']
        stop_loss_price = self.calculate_stop_loss_price(trade_record)
        
        if trade_type == 'buy':
            # 买入交易，当前价格低于止损价格时止损
            return current_price < stop_loss_price
        else:
            # 卖出交易，当前价格高于止损价格时止损
            return current_price > stop_loss_price
    
    def check_position_risk(self, position_value, timestamp=None):
        """
        检查持仓风险
        
        Args:
            position_value: float，当前持仓价值
            timestamp: datetime，时间戳
            
        Returns:
            bool: 是否触发风险警报
            dict: 风险信息
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 更新当前持仓价值
        self.position_risk['current_value'] = position_value
        
        # 更新最大持仓价值
        if position_value > self.position_risk['max_value']:
            self.position_risk['max_value'] = position_value
        
        # 计算回撤
        if self.position_risk['max_value'] > 0:
            drawdown = (self.position_risk['max_value'] - position_value) / self.position_risk['max_value']
            self.position_risk['drawdown'] = drawdown
        else:
            drawdown = 0
        
        # 检查是否触发最大回撤警报
        risk_triggered = drawdown > self.max_drawdown
        
        if risk_triggered and not self.position_risk['stop_loss_triggered']:
            self.position_risk['stop_loss_triggered'] = True
            
            # 记录风险事件
            risk_event = {
                'time': timestamp,
                'type': 'max_drawdown_exceeded',
                'drawdown': drawdown,
                'position_value': position_value,
                'max_position_value': self.position_risk['max_value']
            }
            
            self.risk_events.append(risk_event)
            
            return True, risk_event
        
        return False, None
    
    def correct_failed_trade(self, trade_record, current_price, current_time=None):
        """
        修正失败的交易
        
        Args:
            trade_record: dict，失败的交易记录
            current_price: float，当前价格
            current_time: datetime，当前时间
            
        Returns:
            dict: 修正交易的记录
        """
        if current_time is None:
            current_time = datetime.now()
        
        trade_type = trade_record['type']
        trade_volume = trade_record['volume']
        
        # 执行反向交易进行修正
        correction_type = 'sell' if trade_type == 'buy' else 'buy'
        
        # 记录修正交易
        correction_record = {
            'time': current_time,
            'type': correction_type,
            'price': current_price,
            'volume': trade_volume,
            'value': trade_volume * current_price,
            'is_correction': True,
            'original_trade_time': trade_record['time'],
            'original_trade_price': trade_record['price']
        }
        
        # 记录风险事件
        risk_event = {
            'time': current_time,
            'type': 'trade_correction',
            'original_trade': trade_record,
            'correction_trade': correction_record
        }
        
        self.risk_events.append(risk_event)
        
        return correction_record
    
    def handle_multi_timeframe_failure(self, trade_record, minute_data, five_min_data, current_price, current_time=None):
        """
        处理多时间框架交易失败
        
        按照策略描述，需要结合上一级别的周期判断是否真正失败
        
        Args:
            trade_record: dict，交易记录
            minute_data: DataFrame，分钟级数据
            five_min_data: DataFrame，5分钟级数据
            current_price: float，当前价格
            current_time: datetime，当前时间
            
        Returns:
            dict: 处理结果
        """
        if current_time is None:
            current_time = datetime.now()
        
        trade_time = trade_record['time']
        trade_type = trade_record['type']
        
        # 获取交易后的分钟数据
        post_trade_minute_data = minute_data.loc[minute_data.index > trade_time]
        
        # 获取交易时间所在的5分钟周期结束时间
        trade_minute = trade_time.minute
        next_5min_boundary = trade_time.replace(minute=(trade_minute // 5 + 1) * 5, second=0, microsecond=0)
        
        # 检查5分钟周期是否已经完成
        if current_time < next_5min_boundary:
            # 5分钟周期尚未完成，暂不处理
            return {
                'action': 'wait',
                'reason': 'Waiting for 5-minute cycle to complete',
                'next_check_time': next_5min_boundary
            }
        
        # 获取完成的5分钟周期数据
        completed_5min_data = five_min_data.loc[five_min_data.index <= next_5min_boundary].iloc[-1:]
        
        # 检查5分钟周期是否出现背离
        higher_tf_divergence = False
        if not completed_5min_data.empty:
            if trade_type == 'buy':
                higher_tf_divergence = completed_5min_data['bullish_divergence'].iloc[0]
            else:
                higher_tf_divergence = completed_5min_data['bearish_divergence'].iloc[0]
        
        # 检查分钟级别的背离信号是否消失
        divergence_disappeared = False
        if trade_type == 'buy':
            divergence_disappeared = not post_trade_minute_data['bullish_divergence'].any()
        else:
            divergence_disappeared = not post_trade_minute_data['bearish_divergence'].any()
        
        # 检查价格是否出现相反方向的运动
        price_moved_opposite = False
        if trade_type == 'buy':
            price_moved_opposite = post_trade_minute_data['low'].min() < trade_record['price']
        else:
            price_moved_opposite = post_trade_minute_data['high'].max() > trade_record['price']
        
        # 根据多时间框架判断是否失败
        if price_moved_opposite and divergence_disappeared and not higher_tf_divergence:
            # 满足所有失败条件，执行止损
            correction_record = self.correct_failed_trade(trade_record, current_price, current_time)
            
            return {
                'action': 'stop_loss',
                'reason': 'Trade failed based on multi-timeframe analysis',
                'correction_trade': correction_record
            }
        elif higher_tf_divergence:
            # 5分钟周期出现背离，继续执行原交易方向
            return {
                'action': 'continue',
                'reason': 'Higher timeframe shows divergence, continuing with original trade direction'
            }
        elif not divergence_disappeared:
            # 分钟级别背离信号仍然存在，继续观察
            return {
                'action': 'monitor',
                'reason': 'Minute-level divergence signal still present'
            }
        else:
            # 其他情况，建议谨慎观察
            return {
                'action': 'caution',
                'reason': 'Mixed signals, proceed with caution'
            }
    
    def get_risk_events(self):
        """
        获取风险事件记录
        
        Returns:
            DataFrame: 风险事件记录
        """
        return pd.DataFrame(self.risk_events)
    
    def get_position_risk_status(self):
        """
        获取当前持仓风险状态
        
        Returns:
            dict: 持仓风险状态
        """
        return self.position_risk.copy()
