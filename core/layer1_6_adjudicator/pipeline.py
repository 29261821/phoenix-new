import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import yaml
import copy
from pathlib import Path
import logging

from core.layer1_data.schemas import AssetDataConfig
from core.layer1_data.pipeline import DataLayer
from .schemas import AdjudicatorConfig

from core.system.logger import setup_phoenix_logger
logger = setup_phoenix_logger("L1.6_Adjudicator", "GLOBAL", "ALL")

class FeatureAdjudicator:
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)
            
        l1_6_dict = self.base_config.get("adjudicator_trainer", {})
        self.config = AdjudicatorConfig(**l1_6_dict)
        
        self.asset = self.base_config.get("base_asset", "BTC/USDT")
        self.symbol_safe = self.asset.replace("/", "_")
        self.target_scale = self.base_config.get("base_timeframe", "1m")

    def _load_and_extract_features(self) -> pl.DataFrame:
        logger.info("[*] 讀取並動態生成全高階特徵矩陣...")
        # 修正：Layer 1.6 必須使用專屬的審判庭時段 (adjudicator) 或驗證集 (val)，絕對不能用 train，否則會產生嚴重的特徵過擬合！
        split_dir = Path("infra/feature_store/splits")
        pattern = f"{self.symbol_safe}_{self.target_scale}_adjudicator_*.parquet"
        files = list(split_dir.glob(pattern))
        if not files:
            # Fallback to val
            pattern = f"{self.symbol_safe}_{self.target_scale}_val_*.parquet"
            files = list(split_dir.glob(pattern))
            if not files:
                # 嘗試在上一層找
                files = list(Path(".").glob(f"**/{pattern}"))
                if not files:
                    raise FileNotFoundError(f"找不到審判庭 (adjudicator) 或驗證集 (val) 檔案: {pattern}")
                
        train_file = max(files, key=lambda p: p.stat().st_mtime)
        df_train = pl.read_parquet(train_file)
        
        config_obj = AssetDataConfig(
            base_asset=self.asset,
            target_assets=self.base_config.get("target_assets", {}),
            base_timeframe=self.target_scale,
            macro_timeframes=[],
            warmup_bars=300,
            raw_yaml=self.base_config
        )
        layer1 = DataLayer(config_obj)
        features_dict = layer1.batch(df_train, {}, target_scale=self.target_scale)
        return features_dict[self.target_scale]

    def _quantize_target(self, df: pl.DataFrame) -> pl.DataFrame:
        # 動態計算未來 1-bar 報酬率
        df = df.with_columns(
            ((pl.col("close").shift(-1) / pl.col("close")) - 1.0).alias("target_return_forward")
        )
        # Drop nulls at the end
        df = df.drop_nulls(subset=["target_return_forward"])
        
        # 轉換為 pandas 進行 qcut
        pdf = df.select(["target_return_forward"]).to_pandas()
        pdf["target_class"] = pd.qcut(pdf["target_return_forward"], q=3, labels=[0, 1, 2])
        
        # 修正: Pandas 的 Categorical 無法直接轉入 Polars Series，需要轉成 numpy int32
        target_class_arr = pdf["target_class"].astype(np.int32).values
        return df.with_columns(pl.Series("target_class", target_class_arr).cast(pl.Int32))

    def run(self):
        if not self.config.enabled:
            logger.info("Layer 1.6 特徵審判庭已停用。")
            return
            
        logger.info(f"[*] 啟動 Layer 1.6 特徵審判庭 (模式: {self.config.mode})")
        df = self._load_and_extract_features()
        df = self._quantize_target(df)
        
        exclude_cols = ["datetime", "close_time", "features_mask", "features_reason", "target_return_forward", "target_class"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("_")]
        
        X = df.select(feature_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df.select("target_class").to_numpy().flatten()
        
        n_samples, n_features = X.shape
        real_feature_names = feature_cols
        
        # 1. 影子特徵生成 (Shadow Features)
        if self.config.shadow_rejection:
            logger.info("[*] 執行第一刀：影子特徵檢驗 (Single-Pass Shadow Test)...")
            X_shadow = np.zeros_like(X)
            for i in range(n_features):
                X_shadow[:, i] = np.random.permutation(X[:, i])
                
            X_combined = np.hstack([X, X_shadow])
            shadow_feature_names = [f"SHADOW_{c}" for c in feature_cols]
            all_feature_names = real_feature_names + shadow_feature_names
        else:
            logger.info("[*] 影子特徵檢驗已停用。")
            X_combined = X
            all_feature_names = real_feature_names
            
        # 2. 訓練 LightGBM
        logger.info("[*] 召喚 LightGBM 審判官...")
        params = self.config.training_params.copy()
        params.update({"objective": "multiclass", "num_class": 3, "verbose": -1, "n_jobs": 4})
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_combined, y)
        
        importances = model.feature_importances_
        
        # 3. 影子淘汰判定
        rejected_by_shadow = set()
        feature_scores = {}
        
        if self.config.shadow_rejection:
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]
            shadow_max = np.max(shadow_importances)
            logger.info(f"    - 影子雜訊天花板 (Max Shadow Score): {shadow_max:.2f}")
            
            for i, fname in enumerate(real_feature_names):
                score = real_importances[i]
                feature_scores[fname] = float(score)
                if score <= shadow_max:
                    rejected_by_shadow.add(fname)
            logger.info(f"    - 淘汰了 {len(rejected_by_shadow)}/{n_features} 個連雜訊都不如的特徵。")
        else:
            for i, fname in enumerate(real_feature_names):
                feature_scores[fname] = float(importances[i])
                
        # 4. 正交分群 (Orthogonal Clustering)
        survivors = [f for f in real_feature_names if f not in rejected_by_shadow]
        logger.info(f"[*] 執行第二刀：Spearman 正交分群 (對 {len(survivors)} 個存活特徵)...")
        
        if len(survivors) > 1:
            X_survivors = df.select(survivors).to_numpy()
            X_survivors = np.nan_to_num(X_survivors, nan=0.0)
            
            # 處理全為 0 的欄位，避免 spearmanr 報錯
            corr_matrix, _ = spearmanr(X_survivors, axis=0)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Distance matrix
            dist_matrix = 1 - np.abs(corr_matrix)
            dist_matrix = np.clip(dist_matrix, 0, 1)
            # Ensure symmetry
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
            np.fill_diagonal(dist_matrix, 0)
            
            condensed_dist = squareform(dist_matrix)
            linkage_matrix = hierarchy.linkage(condensed_dist, method='complete')
            
            dist_threshold = 1.0 - self.config.correlation_threshold
            cluster_labels = hierarchy.fcluster(linkage_matrix, dist_threshold, criterion='distance')
            
            cluster_map = {}
            for i, label in enumerate(cluster_labels):
                cluster_map.setdefault(label, []).append(survivors[i])
                
            champions = set()
            redundant = set()
            for label, feats in cluster_map.items():
                # 依分數排序
                feats_sorted = sorted(feats, key=lambda x: feature_scores[x], reverse=True)
                champions.add(feats_sorted[0])
                for f in feats_sorted[1:]:
                    redundant.add(f)
                    
            logger.info(f"    - 劃分為 {len(cluster_map)} 個正交家族。選出 {len(champions)} 個冠軍特徵。")
        else:
            champions = set(survivors)
            redundant = set()
            cluster_map = {1: survivors} if survivors else {}

        # 5. 生成報告與更新 YAML
        report = []
        original_plan = self.base_config.get("meta_features_plan", [])
        new_plan = []
        
        # 建立特徵名稱反查 Indicator 的結構
        indicator_features = {}
        for item in original_plan:
            uid = item["id"]
            indicator_features[uid] = [f for f in real_feature_names if f.startswith(f"{uid}_")]
            
        final_mask = []
            
        for item in original_plan:
            uid = item["id"]
            ind_name = item["indicator"]
            feats = indicator_features[uid]
            
            ind_report = {"indicator": ind_name, "id": uid, "features": []}
            has_survivor = False
            
            for f in feats:
                score = feature_scores.get(f, 0.0)
                if f in rejected_by_shadow:
                    status = "[REJECTED_BY_SHADOW]"
                    cid = -1
                elif f in redundant:
                    status = "[REDUNDANT]"
                    # 找出 cluster id
                    cid = next((k for k, v in cluster_map.items() if f in v), -1)
                else:
                    status = "[CHAMPION]"
                    cid = next((k for k, v in cluster_map.items() if f in v), -1)
                    has_survivor = True
                    final_mask.append(f)
                    
                ind_report["features"].append({
                    "name": f,
                    "score": float(f"{score:.2f}"),
                    "cluster_id": int(cid),
                    "status": status
                })
            
            # Sort features by score in report
            ind_report["features"].sort(key=lambda x: x["score"], reverse=True)
            report.append(ind_report)
            
            if has_survivor or self.config.mode == "semi-auto":
                new_plan.append(item)

        # 6. 寫出檔案
        out_config = copy.deepcopy(self.base_config)
        out_config["adjudicator_report"] = report
        
        if self.config.mode == "auto":
            out_config["meta_features_plan"] = new_plan
            logger.info(f"[*] [AUTO 模式] 物理切除完成！已從 meta_features_plan 剔除 {len(original_plan) - len(new_plan)} 個全軍覆沒的指標。")
            
            # 寫出 features mask
            import json
            mask_path = self.output_dir / f"{self.symbol_safe}_{self.target_scale}_features_mask.json"
            with open(mask_path, "w", encoding="utf-8") as f:
                json.dump(final_mask, f, indent=4)
            logger.info(f"[v] 輸出實盤白名單: {mask_path}")
        else:
            logger.info("[*] [SEMI-AUTO 模式] 僅寫入戰力報告，未修改 meta_features_plan。")
            
        out_yaml_path = self.output_dir / f"{self.symbol_safe}_{self.target_scale}_v6.yaml"
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(out_config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            
        logger.info(f"[v] Layer 1.6 審判庭結案。輸出配置: {out_yaml_path}")
