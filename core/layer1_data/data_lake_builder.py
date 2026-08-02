import polars as pl
import os
import sys
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
import logging
import yaml
import argparse
import importlib

# 確保可以直接在目錄下執行
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [架構隔離防線] 處理 ocean_example 舊程式碼的 sys.modules 污染
ocean_path = Path(project_root) / "zhou_data_engineering" / "ocean_example"
if str(ocean_path) not in sys.path:
    sys.path.insert(0, str(ocean_path))
    
from types import ModuleType
if 'core.layer1_data' not in sys.modules:
    try:
        import features
        import features.Indicators
        import core.layer1_data
        core.layer1_data.features_library = features
        sys.modules['core.layer1_data.features_library'] = features
        sys.modules['core.layer1_data.features_library.Indicators'] = features.Indicators
    except ImportError:
        pass

# [SOTA 防線] 嚴格繼承 Layer 1 的標準元件
from core.layer1_data.processors.market_calendar import MarketCalendar
from core.layer1_data.processors.missing_encoder import MissingnessEncoder
from core.layer1_data.extractors.time_clock import TimeClockExtractor

# 載入專案根目錄的 .env 檔案
load_dotenv(dotenv_path=Path(project_root) / ".env")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLakeBuilder")

class DataLakeBuilder:
    """
    [Layer 1 - SOTA 台股資料湖建構器]
    專注於台股數據管線，在此直接掛載 11 大防線 (三維遺失編碼、時間變形、技術指標)。
    徹底拔除不相關的 Crypto 邏輯，保持架構純淨。
    """
    def __init__(self, data_dir: Optional[str] = None, config_dir: Optional[str] = None):
        if data_dir is None:
            self.data_dir = Path(project_root) / "infra" / "data_lake" / "parquet_store"
        else:
            self.data_dir = Path(data_dir)
            
        if config_dir is None:
            self.config_dir = Path(project_root) / "configs" / "assets"
        else:
            self.config_dir = Path(config_dir)
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.missing_encoder = MissingnessEncoder()

    def load_asset_config(self, asset: str) -> dict:
        """根據商品代號尋找 YAML"""
        symbol_safe = asset.replace("/", "_")
        
        # 1. 直接尋找 symbol_safe.yaml
        config_path = self.config_dir / f"{symbol_safe}.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
                
        # 2. 掃描所有 YAML 尋找 target_assets 內含此 asset 的設定
        yaml_files = list(self.config_dir.glob("*.yaml"))
        for yf in yaml_files:
            with open(yf, "r", encoding="utf-8") as f:
                conf = yaml.safe_load(f)
                if conf and "target_assets" in conf and asset in conf["target_assets"]:
                    return conf
                    
        logger.warning(f"[{asset}] 找不到設定檔。")
        return {"base_timeframe": "15m", "macro_timeframes": ["30m", "60m", "1d"]}

    def _apply_indicators(self, df: pl.DataFrame, timeframe: str, config: dict) -> pl.DataFrame:
        """動態載入 YAML 配置的指標並套用 H 接口與 G 接口"""
        indicator_params = config.get("indicator_default_params", {}).get(timeframe, {})
        meta_plan = config.get("meta_features_plan", [])
        
        exprs = []
        for plan in meta_plan:
            indicator_name = plan["indicator"]
            params = indicator_params.get(indicator_name, {})
            ind_id = plan["id"]
            
            if not params:
                continue
                
            try:
                mod = importlib.import_module(f"features.Indicators.{indicator_name}")
                h_output = None
                # H 接口 (絕對尺度)
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
                raise e
                
        if exprs:
            df = df.with_columns(exprs)
        return df

    def generate_multi_timeframes(self, df_base: pl.DataFrame, base_timeframe: str, macro_timeframes: List[str]) -> Dict[str, pl.DataFrame]:
        """多時間尺度重採樣"""
        df_base = df_base.with_columns(
            (pl.col("datetime").dt.offset_by(base_timeframe) - pl.duration(milliseconds=1)).alias("close_time")
        )
        results = {base_timeframe: df_base}
        
        for tf in macro_timeframes:
            df_tf = df_base.group_by_dynamic("datetime", every=tf).agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("features_mask").any(),
                pl.col("features_reason").max() 
            ])
            df_tf = df_tf.with_columns(
                (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1)).alias("close_time")
            )
            results[tf] = df_tf
            
        return results

    def save_to_datalake(self, asset: str, datasets: Dict[str, pl.DataFrame], sub_dir: str = ""):
        """寫入本地 Parquet"""
        symbol_safe = asset.replace("/", "_")
        asset_dir = self.data_dir / sub_dir / symbol_safe if sub_dir else self.data_dir / symbol_safe
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        for tf, df in datasets.items():
            file_path = asset_dir / f"{symbol_safe}_{tf}.parquet"
            df.write_parquet(str(file_path))
            logger.info(f"[v] 保存完成: {file_path.name} ({len(df)} 筆)")

    async def process_single_asset(self, asset: str, force_rebuild: bool = False):
        """處理單一資產 (台股專用管線)"""
        config = self.load_asset_config(asset)
        base_timeframe = config.get("base_timeframe", "15m")
        macro_timeframes = config.get("macro_timeframes", ["30m", "60m", "1d"])
        
        raw_data_dir = Path(project_root) / "zhou_data_engineering" / "market_data_lake"
        raw_file = raw_data_dir / asset / f"{asset}_TWSE_{base_timeframe}.parquet"
        
        if not raw_file.exists():
            logger.error(f"[{asset}] 找不到台股原始資料 {raw_file}")
            return
            
        logger.info(f"[*] 開始處理台股資產: {asset}")
        df = pl.read_parquet(raw_file)
        if df["datetime"].dtype == pl.Int64:
            df = df.with_columns(pl.from_epoch(pl.col("datetime"), time_unit="ms").alias("datetime"))
            
        # 建立 Market Calendar
        market_cal = MarketCalendar(config.get("market_calendar", {}))
        open_flags = [market_cal.is_market_open(dt) for dt in df["datetime"].to_list()]
        is_open_map = pl.DataFrame({
            "datetime": df["datetime"],
            "is_open": open_flags
        })
        
        # SOTA: 三維遺失編碼
        df_encoded = self.missing_encoder.encode(df, is_open_map)
        
        # 重採樣多尺度
        datasets = self.generate_multi_timeframes(df_encoded, base_timeframe, macro_timeframes)
        
        # 台股交易時間 4.5 小時 = 270 分鐘
        time_clock = TimeClockExtractor(vol_window=30, baseline_window=270)
        
        for tf, df_tf in datasets.items():
            df_tf = time_clock.apply_physical_dt(df_tf)
            df_tf = time_clock.apply_time_deformation(df_tf)
            # 台股開盤 09:00 = 540 分鐘
            df_tf = time_clock.apply_event_clock(df_tf, session_start_mins=540, session_duration_mins=270)
            df_tf = self._apply_indicators(df_tf, tf, config)
            datasets[tf] = df_tf
            
        self.save_to_datalake(asset, datasets, sub_dir="TWSE")

    async def build_multi_asset_lake(self, assets: List[str], force_rebuild: bool = False):
        """多核極限併發"""
        logger.info(f"[🚀] 啟動台股資料湖建構管線: {assets}")
        tasks = [self.process_single_asset(asset, force_rebuild) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for asset, res in zip(assets, results):
            if isinstance(res, Exception):
                logger.error(f"[❌] {asset} 處理失敗，已被隔離。原因: {res}")
        logger.info("[🏁] 所有資產資料湖建構流程完畢！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Data Lake Builder (TWSE)")
    parser.add_argument("--assets", nargs="+", help="指定要抓取的商品清單。若未提供，則自動掃描 configs/assets 下所有 YAML。")
    parser.add_argument("--force-rebuild", action="store_true", help="強制重新編譯。")
    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    builder = DataLakeBuilder()
    
    if args.assets:
        target_assets = args.assets
    else:
        yaml_files = list(builder.config_dir.glob("*.yaml"))
        target_assets = []
        for yf in yaml_files:
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    conf = yaml.safe_load(f)
                    
                    # 過濾非台股設定檔 (例如 crypto)
                    domain = conf.get("domain", "")
                    exchange = conf.get("exchange", "")
                    if domain != "tw_stock" and exchange != "TWSE":
                        continue
                        
                    if conf and "target_assets" in conf:
                        targets = conf["target_assets"]
                        if isinstance(targets, dict):
                            target_assets.extend(list(targets.keys()))
                        elif isinstance(targets, list):
                            target_assets.extend([str(t) for t in targets])
            except Exception as e:
                logger.warning(f"[!] 無法解析 {yf.name}: {e}")
                
        if not target_assets:
            logger.error("[!] 未在 configs/assets 找到任何有效的設定檔。")
            sys.exit(1)
            
    # 去重
    target_assets = list(set(target_assets))
            
    asyncio.run(builder.build_multi_asset_lake(target_assets, force_rebuild=args.force_rebuild))