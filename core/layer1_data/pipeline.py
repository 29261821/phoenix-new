# phoenix/core/layer1_data/pipeline.py

import polars as pl
from typing import Dict, Any, Union
import logging

# 🚨 SOTA 升級：全面改用絕對路徑，根絕 ImportError
from core.layer1_data.schemas import MarketObservation, AssetDataConfig
from core.layer1_data.processors.missing_encoder import MissingnessEncoder
from core.layer1_data.processors.aligner import TimeframeAligner
from core.layer1_data.processors.market_calendar import MarketCalendar
from core.layer1_data.extractors.time_clock import TimeClockExtractor
from core.layer1_data.extractors.feature_proxy import FeatureProxy 

from core.system.logger import setup_phoenix_logger

# 這裡因為 DataLayer 可能是被 L1.5 呼叫，我們先用預設值，或是讓啟動腳本去覆寫它
logger = setup_phoenix_logger("L1_Data", "GLOBAL", "ALL")
class DataLayer:
    """
    [Layer 1 - ⚡ SOTA 協調器]
    負責將原始數據轉化為 MarketObservation，為下游神經網路提供物理一致的數據視角。
    嚴格區分 streaming (實盤逐筆) 與 batch (回測向量化) 雙引擎架構。
    """
    def __init__(self, config: AssetDataConfig):
        self.config = config
        
        # 100分級實例化約束
        self.encoder = MissingnessEncoder()
        self.aligner = TimeframeAligner()
        self.calendar = MarketCalendar()
        self.clock = TimeClockExtractor()
        
        # [防爆紅線] 傳入解析後的 YAML 字典，由 FeatureProxy 動態決定要掛載哪些指標
        self.feature_proxy = FeatureProxy(config.raw_yaml)

    def step(self, base_window: pl.DataFrame, macro_data: Dict[str, pl.DataFrame]) -> MarketObservation:
        """
        [實盤引擎 / Streaming] 全局心跳觸發
        防爆設計：實盤推論時，傳入的 K 線視窗長度必須大於等於 warmup_bars。
        """
        # 🚨 [防爆紅線] 強契約暖機斷言 (Warm-up Assertion)
        current_len = len(base_window)
        if current_len < self.config.warmup_bars:
            raise RuntimeError(
                f"[Layer 1 暖機未完成] 當前視窗長度 {current_len}，"
                f"低於 YAML 要求的最低暖機根數 {self.config.warmup_bars}！請持續累積資料。"
            )

        # 1. 遺失編碼 (Value, Mask, Reason)
        obs_df = self.encoder.encode(base_window)
        
        # 2. 特徵代理注入 (Base Timeframe)
        obs_df = self.feature_proxy.extract_features(obs_df, self.config.base_timeframe)
        
        # 3. 時間語意萃取 (d_tau 與 事件時鐘)
        obs_df = self.clock.extract(obs_df)
        
        # 4. Macro Timeframes 特徵注入
        processed_macro = {}
        for tf, m_df in macro_data.items():
            processed_macro[tf] = self.feature_proxy.extract_features(m_df, tf)
            
        # [SOTA 架構] 廢棄 Flat Wide Table，全面採用多資料流輸出
        latest_obs = obs_df.tail(1)
        
        features_val_dict = {self.config.base_timeframe: latest_obs.select(pl.exclude(["features_mask", "features_reason", "time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"]))}
        features_mask_dict = {self.config.base_timeframe: latest_obs.select(["features_mask"])}
        features_reason_dict = {self.config.base_timeframe: latest_obs.select(["features_reason"])}
        time_features_dict = {self.config.base_timeframe: latest_obs.select(["time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"])}
        
        for tf, m_df in processed_macro.items():
            m_latest = m_df.tail(1)
            features_val_dict[tf] = m_latest.select(pl.exclude(["features_mask", "features_reason"]))
            features_mask_dict[tf] = m_latest.select(["features_mask"])
            features_reason_dict[tf] = m_latest.select(["features_reason"])
        
        return MarketObservation(
            timestamp=int(latest_obs["datetime"][0].timestamp() * 1000),
            asset_id=self.config.base_asset,
            features_value=features_val_dict,
            features_mask=features_mask_dict,
            features_reason=features_reason_dict,
            time_features=time_features_dict
        )

    def batch(self, base_history: pl.DataFrame, macro_history: Dict[str, pl.DataFrame], target_scale: str = None) -> Dict[str, pl.DataFrame]:
        """
        [回測引擎 / Batch] 離線向量化處理
        利用 Polars Lazy Evaluation 榨乾 CPU 效能，並物理性切除不穩定的暖機資料。
        """
        logger.info(f"[*] 啟動 DataLayer Batch Engine: 處理大規模歷史回測資料...")
        
        # 1. 大規模遺失編碼
        df = self.encoder.encode(base_history)
        
        # 2. Base Timeframe 特徵代理注入
        df = self.feature_proxy.extract_features(df, self.config.base_timeframe)
        
        # 3. 向量化時間特徵萃取
        df = self.clock.extract(df)
        
        # 4. Macro Timeframes 特徵代理注入 (動態裁剪，防止 WSL OOM Swap Thrashing)
        processed_macro = {}
        for tf, m_df in macro_history.items():
            if target_scale is not None and target_scale != tf:
                continue
            processed_macro[tf] = self.feature_proxy.extract_features(m_df, tf)
            
        # [SOTA 架構] 廢棄 Flat Wide Table，全面維持多資料流字典
        results = {}
        primary_key = target_scale if target_scale is not None else self.config.base_timeframe
        results[primary_key] = df

        for tf, m_df in processed_macro.items():
            results[tf] = m_df
            
        # 🚨 [防爆紅線] 各尺度切除暖機期 (Warm-up Slicing)
        # 前面 N 根 K 線的 EMA/RSI 等指標處於未收斂狀態 (包含 FFill 的假資料或 NaN)，
        # 必須強制切除，確保送入神經網路的 Batch 張量處於完美的滿秩狀態。
        for tf in results.keys():
            if len(results[tf]) > self.config.warmup_bars:
                results[tf] = results[tf].slice(self.config.warmup_bars)
            else:
                results[tf] = pl.DataFrame(schema=results[tf].schema)
                
        logger.info(f"[v] Batch Engine DL 雙軌輸出完畢，獨立保留 {len(results)} 個時間尺度特徵表。")
        return results