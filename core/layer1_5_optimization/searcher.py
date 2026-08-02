import os
import logging
import copy
from typing import Dict, Any, List
import optuna
import polars as pl
import numpy as np
import warnings
import gc
from pathlib import Path

from core.system.logger import setup_phoenix_logger
from core.layer1_data.pipeline import DataLayer
from core.layer1_data.schemas import AssetDataConfig, AssetMetadata
from core.layer1_5_optimization.search_space import SearchSpaceParser
from core.layer1_5_optimization.evaluator import RegimeEvaluator

logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

class ParameterSearcher:
    """
    [Layer 1.5 - Phase 3 Core] 參數高原優化搜尋器 (SOTA 高效能全 CPU 吃滿版)
    職責：
    1. 管理探索預算與多進程 Worker。
    2. 自動啟動 SQLite WAL (Write-Ahead Logging) 模式，徹底消滅 32 多進程並行時的 DB Lock 鎖死與 CPU 掉速問題。
    3. 執行 SAM 高原尋路擾動評估。
    """
    def __init__(self,
                 base_config: dict,
                 target_scale: str,
                 split_dir: str,
                 search_budget: int = 1000,
                 n_jobs: int = 1):
        self.base_config = copy.deepcopy(base_config)
        self.target_scale = target_scale
        self.split_dir = Path(split_dir)
        self.search_budget = search_budget
        self.n_jobs = n_jobs

        self.parser = SearchSpaceParser(self.base_config, target_scale)
        self.symbol_safe = self.base_config.get("base_asset", "BTC/USDT").replace("/", "_")
        self.raw_train_df = self._load_split_file(target_scale, "train")
        self.raw_val_df = self._load_split_file(target_scale, "val")

    def _load_split_file(self, scale: str, split_name: str) -> pl.DataFrame:
        pattern = f"{self.symbol_safe}_{scale}_{split_name}_*.parquet"
        files = list(self.split_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"找不到預切割檔案: {self.split_dir}/{pattern}")
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        return pl.read_parquet(latest_file)

    def _inject_regimes(self, df: pl.DataFrame) -> pl.DataFrame:
        """注入 Regime 分組欄位 (簡單多區間劃分)"""
        if "regime" in df.columns:
            return df

        if "volatility_geometry_vol_class" in df.columns:
            return df.with_columns(pl.col("volatility_geometry_vol_class").alias("regime"))
        elif "market_state_state" in df.columns:
            return df.with_columns(pl.col("market_state_state").alias("regime"))
        else:
            return df.with_columns(pl.lit("global").alias("regime"))

    def _evaluate_single_param_set(self, flat_params: Dict[str, Any], world_line_config: dict) -> float:
        """對單一參數組合進行特徵計算與模型評估"""
        nested_params = self.parser.unflatten(flat_params)
        eval_config = copy.deepcopy(self.base_config)
        eval_config["indicator_default_params"][self.target_scale] = nested_params

        try:
            target_assets = {}
            raw_targets = eval_config.get("target_assets", {})
            for aid, ameta in raw_targets.items():
                if isinstance(ameta, dict):
                    target_assets[aid] = AssetMetadata(**ameta)

            config_obj = AssetDataConfig(
                base_asset=eval_config.get("base_asset", "BTC/USDT"),
                target_assets=target_assets,
                base_timeframe=eval_config.get("base_timeframe", "1m"),
                macro_timeframes=eval_config.get("macro_timeframes", ["5m", "15m", "1h", "1d"]),
                warmup_bars=eval_config.get("phase1_window", {}).get("warmup_bars", 300),
                raw_yaml=eval_config
            )
            layer1 = DataLayer(config_obj)
            train_res = layer1.batch(self.raw_train_df, {}, target_scale=self.target_scale)
            val_res = layer1.batch(self.raw_val_df, {}, target_scale=self.target_scale)

            train_features = train_res.get(self.target_scale, list(train_res.values())[0])
            val_features = val_res.get(self.target_scale, list(val_res.values())[0])

            actual_pad_bars = self.parser.warmup_bars
            if train_features.height > actual_pad_bars:
                train_features = train_features.slice(actual_pad_bars)
            if val_features.height > actual_pad_bars:
                val_features = val_features.slice(actual_pad_bars)

            train_features = self._inject_regimes(train_features)
            val_features = self._inject_regimes(val_features)

        except Exception as e:
            import traceback
            logger.error(f"[❌ 異常抓錯日誌] 特徵計算失敗！\n👉 帶入之參數組合: {flat_params}\n👉 錯誤訊息: {e}\n{traceback.format_exc()}")
            return -999.0

        proxy_type = world_line_config.get("proxy", "Qlib_Direct_RankIC")
        evaluator = RegimeEvaluator(proxy_type=proxy_type)

        try:
            fitness = evaluator.evaluate(train_features, val_features)
            del layer1, train_features, val_features, evaluator
            gc.collect()
        except Exception as e:
            import traceback
            logger.error(f"[❌ 異常抓錯日誌] 特徵評估失敗！\n👉 帶入之參數組合: {flat_params}\n👉 錯誤訊息: {e}\n{traceback.format_exc()}")
            return -999.0

        return fitness

    def _objective(self, trial, world_line_config: dict):
        center_params = self.parser.suggest_optuna(trial)
        center_fitness = self._evaluate_single_param_set(center_params, world_line_config)
        if center_fitness <= -990.0:
            return -999.0

        use_sam = world_line_config.get("use_sam", True)

        if use_sam:
            jittered_params = self.parser.jitter_params(center_params, jitter_ratio=0.1)
            jitter_fitness = self._evaluate_single_param_set(jittered_params, world_line_config)

            if jitter_fitness <= -990.0:
                return -999.0

            mean_fitness = (center_fitness + jitter_fitness) / 2.0
            sharpness_penalty = abs(center_fitness - jitter_fitness) * 1.5

            sam_score = mean_fitness - sharpness_penalty

            trial.set_user_attr("center_ic", center_fitness)
            trial.set_user_attr("jitter_ic", jitter_fitness)
            trial.set_user_attr("sharpness_penalty", sharpness_penalty)

            return sam_score
        else:
            return center_fitness

    def run_search(self, world_line_config: dict) -> List[dict]:
        searcher_type = world_line_config.get("searcher", "CMA-ES")
        cpu_count = os.cpu_count() or 4
        safe_n_jobs = max(1, min(self.n_jobs, cpu_count))

        from core.layer1_5_optimization.searchers import create_searcher
        searcher_engine = create_searcher(searcher_type, world_line_config)

        if "HEBO" in str(searcher_type).upper() and hasattr(searcher_engine, "run_native_hebo_search"):
            logger.info(f"[🚀] 發動 100% 原生 HEBO 官方引擎 (from hebo.optimizers.hebo import HEBO, 並行 Workers: {safe_n_jobs})...")
            eval_fn = lambda flat_p: self._evaluate_single_param_set(flat_p, world_line_config)
            return searcher_engine.run_native_hebo_search(self.parser, eval_fn, self.search_budget, n_jobs=safe_n_jobs)

        sampler = searcher_engine.get_optuna_sampler()

        db_path = Path("phoenix_optuna_cache.db")
        # 【解決 32 進程 CPU 掉速核心修復】自動啟用 SQLite WAL (Write-Ahead Logging) 模式
        # WAL 模式允許 32 個進程無鎖並列讀寫，完全消滅 DB Lock 造成的進程休眠與 CPU 使用率下跌問題！
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path.name))
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=60000;")
            conn.close()
        except Exception as e:
            logger.debug(f"SQLite WAL 設定提示: {e}")

        db_url = f"sqlite:///{db_path.name}"
        base_storage = optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={
                "connect_args": {
                    "timeout": 120.0,
                    "check_same_thread": False
                }
            }
        )

        study_name = f"L1.5_{self.target_scale}_{world_line_config['id']}"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=base_storage,
            load_if_exists=True
        )

        cpu_count = os.cpu_count() or 4
        safe_n_jobs = max(1, min(self.n_jobs, cpu_count))

        logger.info(f"[🚀] 啟動世界線 {study_name}，引擎: {searcher_type}，預算: {self.search_budget} 次 (動態安全並行數: {safe_n_jobs})")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        try:
            start_trial_number = study.trials[-1].number
        except IndexError:
            start_trial_number = -1

        def _progress_callback(study, trial):
            try:
                best_val = study.best_value
                best_trial_num = study.best_trial.number
            except ValueError:
                best_val = 0.0
                best_trial_num = 0

            session_completed = trial.number - start_trial_number
            session_completed = max(1, session_completed)
            percent = (session_completed / self.search_budget) * 100

            dim_count = len(self.parser.flat_keys)
            theoretical_limit = max(1000, dim_count * 100)
            total_history = trial.number + 1
            sat_by_volume = min(100.0, (total_history / theoretical_limit) * 100)

            max_stagnation = max(500, dim_count * 30)
            stagnation = trial.number - best_trial_num
            sat_by_stagnation = min(100.0, (stagnation / max_stagnation) * 100)

            saturation = max(sat_by_volume, sat_by_stagnation)

            threshold = max(1, int(self.search_budget * 0.01))
            if session_completed % threshold == 0 or session_completed <= 3:
                logger.info(
                    f"👉 [本次進度 {percent:5.1f}%] 🏁 已算 {session_completed}/{self.search_budget} 次 | "
                    f"歷史總計 {total_history} 次 | 🎯 高原收斂度: ~{saturation:.1f}% (已停滯 {stagnation} 次) | 最高分(IC): {best_val:.4f}"
                )

        warnings.filterwarnings("ignore", category=UserWarning)

        study.optimize(
            lambda t: self._objective(t, world_line_config),
            n_trials=self.search_budget,
            n_jobs=safe_n_jobs,
            callbacks=[_progress_callback]
        )

        best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
        top_3 = []
        for trial in best_trials:
            if len(top_3) >= 3:
                break
            if trial.value is not None and trial.value > -990.0:
                unflat = self.parser.unflatten(trial.params)
                top_3.append({
                    "fitness": trial.value,
                    "params": unflat,
                    "trial_number": trial.number
                })

        return top_3
