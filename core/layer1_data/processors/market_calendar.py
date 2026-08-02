import polars as pl
from datetime import datetime, time
import pandas as pd
from typing import Dict, Any

class MarketCalendar:
    """
    [Layer 1 - 防爆核心] 市場日曆引擎 (Futures / Equity Market Calendar)
    職責：針對不同資產的交易所規則進行精確的開收盤判斷，防範非交易時段產生的 Zero-Volatility 幻覺。
    """
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化時傳入從 YAML 解析的 market_calendar 區塊。
        """
        self.config = config_dict
        self.timezone = self.config.get("timezone", "UTC")
        self.holidays = set(self.config.get("holidays", []))
        
        # Parse session times
        self.session_start_str = "09:00"
        self.session_end_str = "13:30"
        
        breaks = self.config.get("session_breaks", [])
        if breaks:
            # For TWSE: session_breaks usually represents the time between close and next open
            # Meaning it closes at break 'start' and opens at break 'end'
            self.close_time_str = breaks[0]["start"]
            self.open_time_str = breaks[0]["end"]
        else:
            self.close_time_str = "23:59"
            self.open_time_str = "00:00"
            
        # Parse weekend closures
        wk = self.config.get("weekend_closure", {})
        self.weekend_start_day = wk.get("start_day", "Friday")
        self.weekend_end_day = wk.get("end_day", "Monday")
        
    def _parse_time(self, t_str: str) -> time:
        h, m = map(int, t_str.split(':'))
        return time(h, m)

    def is_market_open(self, timestamp: datetime) -> bool:
        """
        核心介面：判斷給定時間點 (當地時區) 是否處於交易狀態。
        如果使用 UTC 傳入，外部需要先轉好當地時區，或在此內部轉換 (依架構決定)。
        為求精確與高效，通常要求傳入的 timestamp 已經是該市場的 local time。
        """
        # 1. 檢查假日 (Holiday)
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            return False
            
        # 2. 檢查週末 (Weekend)
        # 台股通常週六 (5)、週日 (6) 休市
        # 若是週五收盤後，或是週一開盤前，也是廣義週末
        wd = timestamp.weekday()
        if wd == 5 or wd == 6: # Saturday or Sunday
            return False
            
        # 3. 檢查每日盤中時段 (Session)
        t = timestamp.time()
        open_t = self._parse_time(self.open_time_str)
        close_t = self._parse_time(self.close_time_str)
        
        if open_t < close_t:
            # 正常日內交易 (如台股 09:00 - 13:30)
            if not (open_t <= t <= close_t):
                return False
        else:
            # 跨日交易 (如 CME 期貨 18:00 - 17:00)
            # 若時間落在 17:00 ~ 18:00 之間則為休市
            if close_t < t < open_t:
                return False
                
        return True