import os
import sys
import yaml
import logging
import polars as pl
from pathlib import Path
from datetime import datetime

# 確保可以直接執行
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [SOTA 防線] 嚴格繼承 Layer 1 的標準元件
from core.layer1_data.processors.market_calendar import MarketCalendar
from core.layer1_data.processors.missing_encoder import MissingnessEncoder
from core.layer1_data.extractors.time_clock import TimeClockExtractor

# [架構隔離防線] 處理 ocean_example 舊程式碼的 sys.modules 污染
# 如此一來不需要竄改 ocean_example，又能完美動態掛載
ocean_path = Path(project_root) / "zhou_data_engineering" / "ocean_example"
if str(ocean_path) not in sys.path:
    sys.path.insert(0, str(ocean_path))
    
# 建立一個假的 core 模組來映射 ocean_example 內硬編碼的路徑
import sys
from types import ModuleType
if 'core.layer1_data' not in sys.modules:
    import features
    core_mod = ModuleType('core')
    layer1_mod = ModuleType('layer1_data')
    core_mod.layer1_data = layer1_mod
    sys.modules['core'] = core_mod
    sys.modules['core.layer1_data'] = layer1_mod
    sys.modules['core.layer1_data.features_library'] = features
    sys.modules['core.layer1_data.features_library.Indicators'] = features.Indicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TWSE_FeatureBuilder")

class TWSEFeatureBuilder:
    """
    [Layer 1 - SOTA 台股特徵管線]
    嚴格遵照覺醒協議 11 大防線：
    - 採用 YAML 單一真理源
    - 嚴禁盲目 ffill 跨越週末，必須套用三維遺失編碼 (Missingness Encoding)
    - 引入時間時鐘與變形係數 (Time Deformation)
    - 保留 H 接口與 G 接口的 Scale-Free 特性
    """
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.market_cal = MarketCalendar(self.config.get("market_calendar", {}))
        self.missing_encoder = MissingnessEncoder()
        
        # 台股交易時間 4.5 小時 = 270 分鐘 (用來計算 Event Clock)
        self.session_duration_mins = 270
        self.time_clock = TimeClockExtractor(vol_window=30, baseline_window=270)
        
        self.raw_data_dir = Path(project_root) / "zhou_data_engineering" / "market_data_lake"
        self.output_dir = Path(project_root) / "infra" / "data_lake" / "parquet_store" / "TWSE"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _apply_indicators(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """動態載入 YAML 配置的指標並套用 H 接口與 G 接口"""
        import importlib
        
        indicator_params = self.config.get("indicator_default_params", {}).get(timeframe, {})
        meta_plan = self.config.get("meta_features_plan", [])
        
        exprs = []
        for plan in meta_plan:
            indicator_name = plan["indicator"]
            params = indicator_params.get(indicator_name, {})
            ind_id = plan["id"]
            
            # [SOTA 缺失防護]: 若 YAML 沒有設定，嚴禁在 Code 中寫預設值
            if not params:
                logger.warning(f"跳過 {indicator_name}: 在 YAML {timeframe} 無配置。")
                continue
                
            try:
                mod = importlib.import_module(f"features.Indicators.{indicator_name}")
                # H 接口 (絕對尺度)
                h_output = None
                if hasattr(mod, "calculate"):
                    h_output = mod.calculate(df, params)
                    if "values" in h_output:
                        for key, expr in h_output["values"].items():
                            exprs.append(expr.alias(f"{ind_id}_{key}"))
                            
                # G 接口 (歸一化特徵)
                adapt_func_name = f"adapt_{indicator_name}"
                if hasattr(mod, adapt_func_name) and h_output:
                    adapt_func = getattr(mod, adapt_func_name)
                    if "close" in df.columns:
                        close_expr = pl.col("close")
                        g_output = adapt_func(h_output, close_expr, params)
                        for key, expr in g_output.items():
                            exprs.append(expr.alias(f"{ind_id}_G_{key}"))
            except Exception as e:
                logger.error(f"[Anti-Evasion Defense] 指標 {indicator_name} 運算失敗: {e}")
                logger.error(f"觸發參數: {params}")
                raise e # 嚴禁吃掉錯誤
                
        if exprs:
            df = df.with_columns(exprs)
            
        return df

    def build_features(self, asset: str, timeframe: str):
        logger.info(f"[*] 啟動 {asset} {timeframe} SOTA 特徵建構 (Layer 1)")
        
        # 1. 讀取原始資料
        raw_file = self.raw_data_dir / asset / f"{asset}_TWSE_{timeframe}.parquet"
        if not raw_file.exists():
            logger.error(f"找不到原始資料 {raw_file}")
            return
            
        df = pl.read_parquet(raw_file)
        
        # 確保 datetime 是毫秒
        if df["datetime"].dtype == pl.Int64:
            df = df.with_columns(pl.from_epoch(pl.col("datetime"), time_unit="ms").alias("datetime"))
            
        # 2. 建立 Market Calendar Mask (判斷每一根 K 線的 local time 是否開盤)
        # 台股資料有時會有跨日/假日的空隙，為了確保遺失編碼準確，先建立連續時間軸 (可選)
        
        # 3. 三維遺失編碼 (Missingness Encoding)
        # SOTA: 我們不提供 map，讓 MissingnessEncoder 自行使用其內部防線填寫 Reason
        # 或更精確地，我們把 is_open 的 map 傳進去
        open_flags = [self.market_cal.is_market_open(dt) for dt in df["datetime"].to_list()]
        is_open_map = pl.DataFrame({
            "datetime": df["datetime"],
            "is_open": open_flags
        })
        df = self.missing_encoder.encode(df, is_open_map)
        
        # 4. 時間時鐘與變形 (Time Deformation)
        df = self.time_clock.apply_physical_dt(df)
        df = self.time_clock.apply_time_deformation(df)
        # Session start for TWSE is 09:00 = 540 mins from midnight
        df = self.time_clock.apply_event_clock(df, session_start_mins=540, session_duration_mins=self.session_duration_mins)
        
        # 5. 指標計算 (動態 YAML 路由)
        df = self._apply_indicators(df, timeframe)
        
        # 6. 輸出至 Layer 1 特徵庫
        out_asset_dir = self.output_dir / asset
        out_asset_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_asset_dir / f"{asset}_{timeframe}.parquet"
        
        df.write_parquet(out_file)
        logger.info(f"[v] {asset} {timeframe} SOTA 特徵產出成功: {out_file}")
        
if __name__ == "__main__":
    config_path = str(Path(project_root) / "configs" / "assets" / "TW_STOCKS.yaml")
    builder = TWSEFeatureBuilder(config_path)
    
    # 測試執行 2330 的 15m, 1d
    for asset in ["2308", "2330", "2454"]:
        for tf in ["15m", "30m", "60m", "1d"]:
            try:
                builder.build_features(asset, tf)
            except Exception as e:
                logger.error(f"[{asset} {tf}] 管線執行失敗: {e}")
