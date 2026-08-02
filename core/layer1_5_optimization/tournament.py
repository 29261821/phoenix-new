import polars as pl
import numpy as np
import scipy.stats as stats
import logging
from core.system.logger import setup_phoenix_logger
from typing import List, Dict, Any, Union
import copy
from pathlib import Path
import yaml
from datetime import datetime
from sklearn.cluster import KMeans

from core.layer1_data.schemas import AssetDataConfig
from core.layer1_data.pipeline import DataLayer
from .evaluator import RegimeEvaluator
from core.layer1_5_optimization.pnl_engine import MultiChartsPnLEngine, PlateauMultiChartsNeighborhoodEngine
from core.layer1_5_optimization.search_space import SearchSpaceParser

logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

class DeathmatchTournament:
    def __init__(self, base_config: dict, target_scale: str, split_dir: str, output_dir: str):
        self.base_config = copy.deepcopy(base_config)
        self.target_scale = target_scale
        self.split_dir = Path(split_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        l1_5_config = self.base_config.get("layer1_5_optimization", {})
        fitness_thresholds_all = l1_5_config.get("fitness_thresholds", {})
        if self.target_scale not in fitness_thresholds_all:
            raise KeyError(f"YAML 缺少 {self.target_scale} 尺度的 fitness_thresholds 配置！請檢查 layer1_5_optimization.fitness_thresholds。")
        self.thresholds = fitness_thresholds_all[self.target_scale]
        self.min_test_psr = float(self.thresholds["min_test_psr"])
        self.max_sharpe_drop = float(self.thresholds["max_sharpe_drop"])

        self.asset = self.base_config.get("base_asset", "BTC/USDT")
        self.symbol_safe = self.asset.replace("/", "_")

        self.raw_train_df = self._load_split_file(self.target_scale, "train")
        self.raw_val_df = self._load_split_file(self.target_scale, "val")
        self.raw_test_df = self._load_split_file(self.target_scale, "test")

        self.regimes_1d_cache = None
        if self.target_scale != "1d":
            self._prepare_1d_regimes()

    def _load_split_file(self, scale: str, split_name: str) -> pl.DataFrame:
        pattern = f"{self.symbol_safe}_{scale}_{split_name}_*.parquet"
        files = list(self.split_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"找不到預切割檔案: {self.split_dir}/{pattern}")
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        return pl.read_parquet(latest_file)

    def _prepare_1d_regimes(self):
        logger.info("[*] 正在為 Regime Anchor 準備 1d 基準特徵...")
        df_1d_train = self._load_split_file("1d", "train")
        df_1d_val = self._load_split_file("1d", "val")
        df_1d_test = self._load_split_file("1d", "test")

        config_obj = AssetDataConfig(
            base_asset=self.asset,
            target_assets=self.base_config.get("target_assets", {}),
            base_timeframe="1d",
            macro_timeframes=[],
            warmup_bars=0,
            raw_yaml=self.base_config,
        )
        layer1_1d = DataLayer(config_obj)

        feat_1d_train = layer1_1d.batch(df_1d_train, {}, target_scale="1d")["1d"]
        feat_1d_val = layer1_1d.batch(df_1d_val, {}, target_scale="1d")["1d"]
        feat_1d_test = layer1_1d.batch(df_1d_test, {}, target_scale="1d")["1d"]

        def extract_features(df):
            atr_cols = [c for c in df.columns if "ATR" in c]
            adx_cols = [c for c in df.columns if "ADX" in c or "Trend_Regime" in c]
            if not atr_cols or not adx_cols:
                df = df.with_columns([
                    pl.col("close").pct_change().rolling_std(14).fill_null(0).alias("_temp_atr"),
                    pl.col("volume").rolling_mean(14).fill_null(0).alias("_temp_vol")
                ])
                return df.select(["_temp_atr", "_temp_vol"]).to_numpy()
            return df.select(atr_cols[:1] + adx_cols[:1]).to_numpy()

        train_feats = np.nan_to_num(extract_features(feat_1d_train))
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(train_feats)

        def assign_regime(df):
            feats = np.nan_to_num(extract_features(df))
            return df.with_columns(
                pl.Series("regime", kmeans.predict(feats)).cast(pl.Int32),
                pl.col("datetime").dt.truncate("1d").alias("_date_key")
            ).select(["_date_key", "regime"])

        r_train = assign_regime(feat_1d_train)
        r_val = assign_regime(feat_1d_val)
        r_test = assign_regime(feat_1d_test)
        self.regimes_1d_cache = pl.concat([r_train, r_val, r_test])

    def _inject_regimes(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.target_scale == "1d":
            feats = np.nan_to_num(df.select([c for c in df.columns if "ATR" in c][:1] + [c for c in df.columns if "ADX" in c or "Trend" in c][:1]).to_numpy())
            if feats.shape[1] == 0:
                feats = np.nan_to_num(df.select([pl.col("close").pct_change().rolling_std(14).fill_null(0)]).to_numpy())
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            return df.with_columns(pl.Series("regime", kmeans.fit_predict(feats)).cast(pl.Int32))
        else:
            temp_df = df.with_columns(pl.col("datetime").dt.truncate("1d").alias("_date_key"))
            temp_df = temp_df.join(self.regimes_1d_cache, on="_date_key", how="left")
            return temp_df.drop("_date_key").with_columns(pl.col("regime").fill_null(strategy="forward").fill_null(0))

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        positions = np.sign(y_pred)
        strategy_returns = positions * y_true
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        sharpe = 0.0
        if std_ret > 1e-8:
            sharpe = mean_ret / std_ret
        N = len(strategy_returns)
        skew = stats.skew(strategy_returns)
        kurt = stats.kurtosis(strategy_returns, fisher=False)
        sr_star = 0.0
        if std_ret > 1e-8:
            denominator = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * (sharpe ** 2))
            if denominator > 0:
                psr = stats.norm.cdf((sharpe - sr_star) * np.sqrt(N - 1) / denominator)
            else:
                psr = 0.0
        else:
            psr = 0.0
        if std_ret > 1e-8 and 'denominator' in locals() and denominator > 0:
            psr_z = (sharpe - sr_star) * np.sqrt(N - 1) / denominator
        else:
            psr_z = 0.0
        return {"sharpe": sharpe, "psr_z": psr_z, "max_dd": -0.10}

    def evaluate_and_save(self, top_candidates: List[dict], world_line_id: str, output_path: Union[str, Path]):
        output_path = Path(output_path)
        logger.info(f"[*] 進入 Phase 4 死亡競技場：解封 Test Set 測試 {world_line_id} 的 Top 3 菁英...")
        results = []
        for idx, candidate in enumerate(top_candidates):
            trial_config = copy.deepcopy(self.base_config)
            parser = SearchSpaceParser(self.base_config, self.target_scale)
            # 必須解展平 (Unflatten) 展平字典，還原為 nested dict 給 FeatureProxy
            nested_params = parser.unflatten(candidate["params"]) if isinstance(candidate["params"], dict) else candidate["params"]
            trial_config["indicator_default_params"][self.target_scale] = nested_params

            actual_pad_bars = self.base_config.get("phase1_window", {}).get("warmup_bars", 300)
            config_obj = AssetDataConfig(
                base_asset=self.asset,
                target_assets=trial_config.get("target_assets", {}),
                base_timeframe=self.target_scale,
                macro_timeframes=[],
                warmup_bars=actual_pad_bars,
                raw_yaml=trial_config,
            )

            layer1 = DataLayer(config_obj)
            train_res = layer1.batch(self.raw_train_df, {}, target_scale=self.target_scale)
            val_res = layer1.batch(self.raw_val_df, {}, target_scale=self.target_scale)
            test_res = layer1.batch(self.raw_test_df, {}, target_scale=self.target_scale)

            train_features = train_res.get(self.target_scale, list(train_res.values())[0])
            val_features = val_res.get(self.target_scale, list(val_res.values())[0])
            test_features = test_res.get(self.target_scale, list(test_res.values())[0])

            # 核心機轉對齊：若 DataLayer 未切除，手動切除前 actual_pad_bars 根 Padding
            if len(train_features) == self.raw_train_df.height and len(train_features) > actual_pad_bars:
                train_features = train_features.slice(actual_pad_bars)
            if len(val_features) == self.raw_val_df.height and len(val_features) > actual_pad_bars:
                val_features = val_features.slice(actual_pad_bars)
            if len(test_features) == self.raw_test_df.height and len(test_features) > actual_pad_bars:
                test_features = test_features.slice(actual_pad_bars)

            train_features = self._inject_regimes(train_features)
            val_features = self._inject_regimes(val_features)
            test_features = self._inject_regimes(test_features)

            l1_5_config = self.base_config.get("layer1_5_optimization", {})
            world_lines = l1_5_config.get("world_lines", [])
            proxy_type = "LightGBM"
            for wl in world_lines:
                if wl["id"] == world_line_id:
                    proxy_type = wl.get("proxy", "LightGBM")
                    break

            evaluator = RegimeEvaluator(proxy_type=proxy_type)
            pnl_engine = MultiChartsPnLEngine(self.base_config.get("trading_economics", {}))
            plateau_engine = PlateauMultiChartsNeighborhoodEngine(pnl_engine, n_neighbors=5, perturb_ratio=0.10)
            parser = SearchSpaceParser(self.base_config, self.target_scale)

            is_direct_ic = any(kw in str(proxy_type).upper() for kw in ["RANKIC", "DIRECT", "ALLFEATURE", "IC"])
            if is_direct_ic:
                val_fitness = evaluator.evaluate(train_features, val_features)
                test_fitness = evaluator.evaluate(val_features, test_features)

                val_score = max(0.01, val_fitness)
                test_score = max(0.0, test_fitness)
                sharpe_drop = (test_fitness - val_fitness) / (abs(val_fitness) + 1e-8)
                psr_z = test_fitness / 5.0

                # 構建全指標組合訊號進行高原損益模擬
                _, _, _, feature_cols = evaluator._prepare_data(test_features)

                mc_report, plateau_profile = plateau_engine.evaluate_plateau(
                    df=test_features,
                    feature_cols=feature_cols,
                    parser=parser,
                    center_params=candidate["params"]
                )

                candidate["test_psr_z"] = psr_z
                candidate["sharpe_drop"] = sharpe_drop
                candidate["test_max_dd"] = -abs(mc_report["Max Strategy Drawdown (%)"] / 100.0)
                candidate["test_fitness"] = test_fitness
                candidate["multicharts_report"] = mc_report
                candidate["plateau_topology_profile"] = plateau_profile
                results.append(candidate)
            else:
                X_train, y_train, _, _ = evaluator._prepare_data(train_features)
                X_val, y_val, _, _ = evaluator._prepare_data(val_features)
                X_test, y_test, _, feature_cols = evaluator._prepare_data(test_features)

                is_lgbm = "LIGHTGBM" in str(proxy_type).upper() or "LGB" in str(proxy_type).upper()
                if is_lgbm:
                    import lightgbm as lgb
                    import os
                    lgb_threads = int(os.environ.get("POLARS_MAX_THREADS", 1))
                    sample_size = len(X_train)
                    if sample_size < 1000:
                        leaves, estimators = 3, 30
                    elif sample_size < 10000:
                        leaves, estimators = 7, 50
                    else:
                        leaves, estimators = 15, 100
                    model = lgb.LGBMRegressor(
                        n_estimators=estimators,
                        learning_rate=0.1,
                        num_leaves=leaves,
                        random_state=42,
                        n_jobs=lgb_threads,
                        verbose=-1,
                    )
                else:
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0, random_state=42)

                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                test_preds = model.predict(X_test)

                val_metrics = self._calculate_metrics(y_val, val_preds)
                test_metrics = self._calculate_metrics(y_test, test_preds)

                val_sr = val_metrics["sharpe"]
                test_sr = test_metrics["sharpe"]
                sharpe_drop = (test_sr - val_sr) / (abs(val_sr) + 1e-8)

                # 高原微區損益模擬
                mc_report, plateau_profile = plateau_engine.evaluate_plateau(
                    df=test_features,
                    feature_cols=feature_cols,
                    parser=parser,
                    center_params=candidate["params"]
                )

                candidate["test_psr_z"] = test_metrics["psr_z"]
                candidate["sharpe_drop"] = sharpe_drop
                candidate["test_max_dd"] = -abs(mc_report["Max Strategy Drawdown (%)"] / 100.0)
                candidate["multicharts_report"] = mc_report
                candidate["plateau_topology_profile"] = plateau_profile
                results.append(candidate)

        passed = [r for r in results if r["test_psr_z"] >= self.min_test_psr and r["sharpe_drop"] >= -abs(self.max_sharpe_drop)]
        if passed:
            passed.sort(key=lambda x: x["test_psr_z"], reverse=True)
            best_candidate = passed[0]
            logger.info(f"[🏆] 結局 A (達標)：{world_line_id} 選出最強王者 (PSR={best_candidate['test_psr_z']:.2f})！")
            self._stamp_and_export(best_candidate, world_line_id, "PASSED", output_path)
        else:
            results.sort(key=lambda x: x.get("fitness", x.get("val_fitness", 0)), reverse=True)
            best_loser = results[0]
            logger.warning(f"[💀] 結局 B (全軍覆沒)：無人通過 Test 門檻 (PSR >= {self.min_test_psr} 且跌幅不超 {self.max_sharpe_drop})。強制作出 Top 1 供除錯。")
            self._stamp_and_export(best_loser, world_line_id, "FAILED_THRESHOLDS", output_path)
        return output_path

    def _stamp_and_export(self, candidate: dict, wl_id: str, status: str, output_path: Path):
        export_config = copy.deepcopy(self.base_config)
        parser = SearchSpaceParser(self.base_config, self.target_scale)
        nested_params = parser.unflatten(candidate["params"]) if isinstance(candidate["params"], dict) else candidate["params"]
        export_config["indicator_default_params"][self.target_scale] = nested_params
        
        mc_report = candidate.get("multicharts_report", {})
        plateau_profile = candidate.get("plateau_topology_profile", {})
        export_config["_metadata"] = {
            "optimizer": wl_id,
            "status": status,
            "performance": {
                "val_fitness": float(candidate.get("fitness", candidate.get("val_fitness", 0))),
                "test_psr": float(candidate.get("test_psr_z", 0.0)),
                "sharpe_drop": f"{candidate.get('sharpe_drop', 0.0)*100:.2f}%",
                "test_max_dd": f"{candidate.get('test_max_dd', 0.0)*100:.2f}%",
            },
            "plateau_topology_profile": plateau_profile,
            "multicharts_performance_summary": mc_report
        }
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(export_config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        logger.info(f"[v] 烙印完成: {output_path.name} (Status: {status})")
