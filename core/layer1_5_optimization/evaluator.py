import polars as pl
import numpy as np

# [環境熱修復] WSL conda 下某些套件的 metadata 可能毀損導致 cupy 崩潰 (LightGBM 依賴 Dask 依賴 Cupy)
import importlib.metadata
_original_distributions = importlib.metadata.distributions

def _safe_distributions():
    for d in _original_distributions():
        if d.metadata is not None:
            yield d
        else:
            try:
                print(f"\n[⚠️ 警告] 發現損壞的 Python 套件！路徑: {getattr(d, '_path', 'Unknown')}")
                print("[⚠️ 警告] 該套件的 metadata 已經損毀，這會導致 cupy/dask 等套件崩潰。")
                print("[⚠️ 警告] 建議您在終端機中刪除上述資料夾以永久修復環境。\n")
            except Exception:
                pass
importlib.metadata.distributions = _safe_distributions

import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import logging
from core.system.logger import setup_phoenix_logger

logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

class RegimeEvaluator:
    """
    [Layer 1.5 - Phase 2] SOTA 全指標無盲點評估器 (Qlib Direct Multi-Factor Rank IC & IR)
    職責：
    1. 接收 Train/Val 的記憶體 DataFrame (包含 features, target, regime)。
    2. 提供 Qlib_Direct_RankIC 評估模式：
       - 直接計算 31 個指標欄位對目標報酬率的 Rank IC 與信息比率 (Information Ratio, IR)。
       - 100% 保證每一個指標參數變動均能直接反應在適應度上，完全消滅樹模型忽視 25 個指標的零梯度死高原！
    3. 同時支援 Ridge 與 LightGBM 作為可選模型。
    """
    def __init__(self, proxy_type: str = "Qlib_Direct_RankIC"):
        self.proxy_type = proxy_type
        self.epsilon = 1e-8 # 避免除以 0

    def _prepare_data(self, df: pl.DataFrame):
        """將 DataFrame 轉換為機器學習與 Rank IC 計算所需的格式"""
        target_col = [c for c in df.columns if c.startswith("target_return_")]
        if not target_col:
            df = df.with_columns(
                ((pl.col("close").shift(-1) / pl.col("close")) - 1.0).alias("target_return_forward")
            )
            target_col = ["target_return_forward"]

        target_col = target_col[0]

        exclude_cols = ["datetime", "close_time", "features_mask", "features_reason", "regime", target_col]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df.select(feature_cols).to_numpy()
        y = df.select(target_col).to_numpy().flatten()
        regimes = df.select("regime").to_numpy().flatten()

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, regimes, feature_cols

    def evaluate(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> float:
        """
        核心評估邏輯：回傳 Qlib 全指標 Rank IC / IR 或模型評估分數
        """
        # 1. 優先處理解耦全指標直算模式 (自動比對 RankIC, Direct, AllFeature, IC 關鍵字，未來擴充名稱無需修改代碼)
        is_direct_ic = any(kw in str(self.proxy_type).upper() for kw in ["RANKIC", "DIRECT", "ALLFEATURE", "IC"])
        if is_direct_ic:
            _, y_val, val_regimes, feature_cols = self._prepare_data(val_df)
            if len(feature_cols) == 0 or len(y_val) < 30:
                return 0.0

            unique_regimes = np.unique(val_regimes)
            regime_scores = []

            for r in unique_regimes:
                mask = (val_regimes == r)
                y_true_r = y_val[mask]
                if len(y_true_r) < 20:
                    continue

                sub_df = val_df.filter(pl.col("regime") == r)
                feature_ics = []

                for col in feature_cols:
                    x_col = sub_df.select(col).to_numpy().flatten()
                    x_col = np.nan_to_num(x_col, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.std(x_col) < 1e-12:
                        continue
                    ic, _ = spearmanr(x_col, y_true_r)
                    if not np.isnan(ic):
                        feature_ics.append(abs(ic))

                if feature_ics:
                    mean_ic = float(np.mean(feature_ics))
                    std_ic = float(np.std(feature_ics))
                    # 信息比率 Rank IR = Mean IC / (Std IC + eps)
                    rank_ir = mean_ic / (std_ic + 1e-5)

                    # 計算指標間的共線冗餘懲罰 (Collinearity Penalty)
                    collinear_penalty = 0.0
                    if len(feature_cols) > 1:
                        try:
                            feat_mat = sub_df.select(feature_cols).to_numpy()
                            feat_mat = np.nan_to_num(feat_mat, nan=0.0)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr_matrix = np.abs(np.corrcoef(feat_mat, rowvar=False))
                            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                            np.fill_diagonal(corr_matrix, 0.0)
                            collinear_penalty = float(np.nanmean(corr_matrix))
                        except Exception:
                            collinear_penalty = 0.0

                    # 終極 SOTA 直算適應度：Rank IR * 10.0 + Mean IC * 100.0 - 冗餘共線性懲罰
                    fitness_score = rank_ir * 10.0 + mean_ic * 100.0 - collinear_penalty * 2.0
                    regime_scores.append(fitness_score)

            if not regime_scores:
                return 0.0

            return float(np.mean(regime_scores))

        # 2. 模型代理模式 (LightGBM / Ridge / 降級彈性配對)
        X_train, y_train, _, _ = self._prepare_data(train_df)
        X_val, y_val, val_regimes, _ = self._prepare_data(val_df)

        is_lgbm = "LIGHTGBM" in str(self.proxy_type).upper() or "LGB" in str(self.proxy_type).upper()
        if is_lgbm:
            import os
            lgb_threads = int(os.environ.get("POLARS_MAX_THREADS", 1))
            sample_size = len(X_train)
            if sample_size < 1000:
                leaves = 7
                estimators = 50
            elif sample_size < 10000:
                leaves = 15
                estimators = 100
            else:
                leaves = 31
                estimators = 150

            model = lgb.LGBMRegressor(
                n_estimators=estimators,
                learning_rate=0.05,
                num_leaves=leaves,
                random_state=42,
                n_jobs=lgb_threads,
                verbose=-1
            )
            model.fit(X_train, y_train)
        else:
            # 彈性備用模型：預設採用線性 Ridge 回歸，相容未來所有自訂模型名稱
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)

        preds = model.predict(X_val)

        unique_regimes = np.unique(val_regimes)
        regime_ics = []

        for r in unique_regimes:
            mask = (val_regimes == r)
            y_true_r = y_val[mask]
            y_pred_r = preds[mask]
            if len(y_true_r) < 30:
                continue

            ic, _ = spearmanr(y_true_r, y_pred_r)
            if not np.isnan(ic):
                regime_ics.append(ic)

        if not regime_ics:
            return 0.0

        mean_ic = float(np.mean(regime_ics))
        std_ic = float(np.std(regime_ics))
        ir = mean_ic / (std_ic + 1e-5)

        return float(mean_ic * 100.0 + ir)
