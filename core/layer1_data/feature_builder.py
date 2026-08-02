import argparse
import polars as pl
import os
import asyncio
from pathlib import Path
import logging
import yaml
import sys
from typing import Dict, Union

# 確保可以直接在目錄下執行
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.layer1_data.schemas import AssetDataConfig
from core.layer1_data.pipeline import DataLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureBuilder")

class FeatureBuilder:
    """
    [Layer 1 - 終極兵器] Feature Store Builder
    負責從 Data Lake 讀取原始 K 線，透過 DataLayer (FeatureProxy + Aligner)
    產出神經網路/樹模型專用的高維度特徵寬表。
    """
    def __init__(self, data_lake_dir: str = None, feature_store_dir: str = None, config_dir: str = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        
        self.data_lake_dir = Path(data_lake_dir) if data_lake_dir else project_root / "infra" / "data_lake" / "parquet_store"
        self.feature_store_dir = Path(feature_store_dir) if feature_store_dir else project_root / "infra" / "feature_store"
        self.config_dir = Path(config_dir) if config_dir else project_root / "configs" / "assets"
        
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, asset: str, base_timeframe: str, macro_timeframes: list) -> tuple:
        """從 Data Lake 載入已經清理過的原始資料"""
        symbol_safe = asset.replace("/", "_")
        asset_lake_dir = self.data_lake_dir / symbol_safe
        
        base_path = asset_lake_dir / f"{symbol_safe}_{base_timeframe}.parquet"
        if not base_path.exists():
            raise FileNotFoundError(f"找不到 Base 資料檔: {base_path}，請先執行 data_lake_builder.py")
            
        base_df = pl.read_parquet(base_path)
        
        macro_dict = {}
        for tf in macro_timeframes:
            macro_path = asset_lake_dir / f"{symbol_safe}_{tf}.parquet"
            if macro_path.exists():
                macro_dict[tf] = pl.read_parquet(macro_path)
            else:
                logger.warning(f"[{asset}] 找不到 Macro 資料檔: {macro_path}，將略過該尺度。")
                
        return base_df, macro_dict

    def process_single_asset(self, asset: str):
        logger.info(f"[*] 開始為 {asset} 建構 Feature Store...")
        
        # 1. 讀取並解析 YAML
        symbol_safe = asset.replace("/", "_")
        config_path = self.config_dir / f"{symbol_safe}.yaml"
        if not config_path.exists():
            logger.error(f"[{asset}] 找不到設定檔 {config_path}")
            return
            
        with open(config_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
            
        config = AssetDataConfig(
            base_asset=asset,
            target_assets=raw_yaml.get("target_assets", {}),
            base_timeframe=raw_yaml.get("base_timeframe", "1m"),
            macro_timeframes=raw_yaml.get("macro_timeframes", ["5m", "15m", "1h", "1d"]),
            warmup_bars=raw_yaml.get("phase1_window", {}).get("warmup_bars", 300),
            raw_yaml=raw_yaml
        )
        
        # 2. 載入 Data Lake 原始資料
        try:
            base_df, macro_dict = self.load_raw_data(asset, config.base_timeframe, config.macro_timeframes)
        except Exception as e:
            logger.error(f"[{asset}] 載入原始資料失敗: {e}")
            return
            
        # 3. 實例化 DataLayer 並進行 Batch 運算
        layer1 = DataLayer(config)
        features_output = layer1.batch(base_df, macro_dict)
        
        # 4. 儲存至 Feature Store (純粹多資料流)
        asset_feature_dir = self.feature_store_dir / symbol_safe
        asset_feature_dir.mkdir(parents=True, exist_ok=True)
        
        # SOTA 修正：Layer 1 全面廢棄 Flat Wide Table，只輸出獨立的 Parquet
        for tf, df in features_output.items():
            out_path = asset_feature_dir / f"{symbol_safe}_features_{tf}.parquet"
            df.write_parquet(out_path)
            logger.info(f"[v] {asset} {tf} 儲存成功: {out_path.name} (寬度: {len(df.columns)}, 筆數: {len(df)})")

    def build_all(self, target_assets: list = None):
        """全量建構"""
        if not target_assets:
            yaml_files = list(self.config_dir.glob("*.yaml"))
            target_assets = []
            for yf in yaml_files:
                try:
                    with open(yf, "r", encoding="utf-8") as f:
                        conf = yaml.safe_load(f)
                        if conf and "base_asset" in conf:
                            target_assets.append(conf["base_asset"])
                except Exception as e:
                    pass
                    
        for asset in target_assets:
            try:
                self.process_single_asset(asset)
            except Exception as e:
                logger.error(f"[{asset}] Feature Store 建構發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Feature Store Builder (Layer 1 Final)")
    parser.add_argument("--assets", nargs="+", help="指定要計算特徵的商品 (例如 BTC/USDT)。未提供則全掃描。")
    args = parser.parse_args()

    builder = FeatureBuilder()
    builder.build_all(target_assets=args.assets)
