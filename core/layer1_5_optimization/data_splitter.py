import polars as pl
import logging
from core.system.logger import setup_phoenix_logger
from sklearn.cluster import KMeans
from datetime import datetime, timezone
import yaml
from pathlib import Path

logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

class DataSplitter:
    """
    [Layer 1.5 - Phase 1] 物理切割與體制標註 (Physical Split & Regime Annotation)
    職責：
    1. 依照 YAML 中各尺度的絕對時間牆 (data_boundaries) 進行 Train/Val/Test 切割，杜絕未來函數。
    2. 基於日線 (1d) 的 ATR 與 ADX 進行 K-Means 體制標註 (只在 Train Set fit，杜絕外洩)。
    3. 將日線的體制標籤廣播 (Broadcast) 到其他微觀尺度 (1m, 5m 等)。
    """
    def __init__(self, config_path: Path, feature_store_dir: Path):
        self.config_path = config_path
        self.feature_store_dir = feature_store_dir
        self.config = self._load_config()
        self.kmeans_model = None
        self.df_1d_regimes_cache = None

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_boundary(self, date_str: str) -> datetime:
        """解析 YYYY-MM-DD 為 datetime (Timezone-naive, 以配合 DataFrame 的 datetime[ms])"""
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _get_parquet_path(self, asset: str, tf: str) -> Path:
        symbol_safe = asset.replace("/", "_")
        return self.feature_store_dir / symbol_safe / f"{symbol_safe}_features_{tf}.parquet"

    def _get_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """動態提取 ATR 與 ADX 作為 K-Means 的特徵"""
        # 尋找名稱包含 ATR 與 ADX 的欄位
        atr_cols = [c for c in df.columns if "ATR" in c]
        adx_cols = [c for c in df.columns if "ADX" in c or "Trend_Regime" in c]
        
        # 降落傘：如果找不到，就找 close 跟 volume 的變異數作為替代
        if not atr_cols or not adx_cols:
            logger.warning("在特徵中找不到標準的 ATR/ADX，將使用基礎價量計算替代體制特徵。")
            df = df.with_columns([
                pl.col("close").pct_change().rolling_std(14).fill_null(0).alias("_temp_atr"),
                pl.col("volume").rolling_mean(14).fill_null(0).alias("_temp_vol")
            ])
            return df.select(["_temp_atr", "_temp_vol"])
            
        selected_cols = atr_cols[:1] + adx_cols[:1] # 只取最具代表性的一個
        return df.select(selected_cols)

    def _split_by_boundaries(self, df: pl.DataFrame, boundaries: dict) -> dict:
        """依照邊界精準切割 Train/Val/Test"""
        train_start = self._parse_boundary(boundaries["train"][0])
        train_end = self._parse_boundary(boundaries["train"][1])
        val_start = self._parse_boundary(boundaries["val"][0])
        val_end = self._parse_boundary(boundaries["val"][1])
        test_start = self._parse_boundary(boundaries["test"][0])
        test_end = self._parse_boundary(boundaries["test"][1])

        # 這裡必須轉為 ms epoch 進行比對 (因為 feature_builder 的 datetime 可能是 dt 或是 ms)
        train_df = df.filter((pl.col("datetime") >= train_start) & (pl.col("datetime") <= train_end))
        val_df = df.filter((pl.col("datetime") >= val_start) & (pl.col("datetime") <= val_end))
        test_df = df.filter((pl.col("datetime") >= test_start) & (pl.col("datetime") <= test_end))

        return {"train": train_df, "val": val_df, "test": test_df}

    def _annotate_regimes_1d(self, asset: str) -> dict:
        """
        處理 1d 尺度，訓練 K-Means (僅限 Train)，並預測出體制標籤。
        """
        l1_5_config = self.config.get("layer1_5_optimization", {})
        bounds_1d = l1_5_config.get("data_boundaries", {}).get("1d")
        if not bounds_1d:
            raise ValueError(f"YAML 缺少 1d 的 data_boundaries 隔離牆設定！")

        path_1d = self._get_parquet_path(asset, "1d")
        if not path_1d.exists():
            raise FileNotFoundError(f"找不到 1d 基準特徵檔: {path_1d}")

        df_1d = pl.read_parquet(path_1d)
        splits = self._split_by_boundaries(df_1d, bounds_1d)

        # 1. 萃取 K-Means 所需特徵
        train_features = self._get_regime_features(splits["train"]).to_numpy()
        val_features = self._get_regime_features(splits["val"]).to_numpy()
        test_features = self._get_regime_features(splits["test"]).to_numpy()

        # 2. 訓練 K-Means (嚴格限制在 Train Set 杜絕外洩)
        logger.info("[*] 正在 Train Set 上訓練 1d K-Means(k=4) 體制錨點...")
        self.kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        
        # 由於可能有 NaN (起步階段)，需補 0
        import numpy as np
        train_features = np.nan_to_num(train_features)
        val_features = np.nan_to_num(val_features)
        test_features = np.nan_to_num(test_features)

        self.kmeans_model.fit(train_features)

        # 3. 預測體制並賦予標籤
        splits["train"] = splits["train"].with_columns(pl.Series("regime", self.kmeans_model.predict(train_features)).cast(pl.Int32))
        splits["val"] = splits["val"].with_columns(pl.Series("regime", self.kmeans_model.predict(val_features)).cast(pl.Int32))
        splits["test"] = splits["test"].with_columns(pl.Series("regime", self.kmeans_model.predict(test_features)).cast(pl.Int32))

        return splits

    def get_split(self, asset: str, target_scale: str) -> dict:
        """
        主介面：取得目標尺度切好的 Train/Val/Test，並附帶對齊好的 Regime 標籤。
        """
        logger.info(f"[*] 啟動 Phase 1: 切割 {asset} {target_scale} 並注入體制標籤...")
        
        # 1. 先取得 1d 的體制標籤 (Regime Anchor)
        splits_1d = self._annotate_regimes_1d(asset)

        # 無論目標尺度為何，都強制建立 df_1d_regimes_cache，供後續在記憶體中做 Regime 廣播
        df_1d_all = pl.concat([splits_1d["train"], splits_1d["val"], splits_1d["test"]])
        self.df_1d_regimes_cache = df_1d_all.select([
            pl.col("datetime").dt.truncate("1d").alias("_date_key"),
            pl.col("regime")
        ])

        # 如果目標尺度剛好就是 1d，直接回傳
        if target_scale == "1d":
            return splits_1d
            
        # 2. 如果目標尺度是微觀尺度 (e.g. 1m)，我們需要廣播體制
        l1_5_config = self.config.get("layer1_5_optimization", {})
        bounds_target = l1_5_config.get("data_boundaries", {}).get(target_scale)
        if not bounds_target:
            raise ValueError(f"YAML 缺少 {target_scale} 的 data_boundaries 隔離牆設定！")

        path_target = self._get_parquet_path(asset, target_scale)
        if not path_target.exists():
            raise FileNotFoundError(f"找不到 {target_scale} 基準特徵檔: {path_target}")

        df_target = pl.read_parquet(path_target)
        splits_target = self._split_by_boundaries(df_target, bounds_target)

        # 對微觀尺度進行 Broadcast (Forward-Fill Join)
        for ds in ["train", "val", "test"]:
            # 將微觀的時間也截斷到日，以便跟 1d 對齊
            temp_df = splits_target[ds].with_columns(
                pl.col("datetime").dt.truncate("1d").alias("_date_key")
            )
            # 進行左連線
            temp_df = temp_df.join(self.df_1d_regimes_cache, on="_date_key", how="left")
            # 清除暫存鍵，並對週末/休市導致的空洞進行向下填補 (forward fill)，再補預設值 0
            splits_target[ds] = temp_df.drop("_date_key").with_columns(
                pl.col("regime").fill_null(strategy="forward").fill_null(0)
            )

        logger.info(f"[v] Phase 1 完工！成功將 1d 體制廣播至 {target_scale}。")
        return splits_target
