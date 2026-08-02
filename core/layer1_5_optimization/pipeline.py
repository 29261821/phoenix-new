import argparse
import logging
import os
import re
import sys
from pathlib import Path
import yaml

from core.system.logger import setup_phoenix_logger
from core.layer1_5_optimization.searcher import ParameterSearcher
from core.layer1_5_optimization.tournament import DeathmatchTournament

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

def auto_generate_output_name(config_path: Path, target_scale: str, world_line: str, out_dir: Path) -> Path:
    """自動推導接力輸出的 YAML 名稱"""
    base_name = config_path.stem
    if world_line not in base_name:
        new_name = f"{base_name}_{world_line}_{target_scale}"
    else:
        new_name = f"{base_name}_{target_scale}"

    scales = ["1m", "5m", "15m", "1h", "1d"]
    count = sum(1 for s in scales if f"_{s}" in new_name or new_name.endswith(s))

    if count >= 5:
        clean_name = new_name
        for s in scales:
            clean_name = clean_name.replace(f"_{s}", "")

        v_match = re.search(r'_v(\d+)', clean_name)
        if v_match:
            v_num = int(v_match.group(1))
            clean_name = re.sub(r'_v\d+', f'_v{v_num+1}', clean_name)
        else:
            clean_name = f"{clean_name}_v1"

        return out_dir / f"{clean_name}.yaml"

    return out_dir / f"{new_name}.yaml"


class OptimizationPipeline:
    def __init__(self, config_path: str, output_path: str, target_scale: str, split_dir: str):
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        self.target_scale = target_scale
        self.split_dir = Path(split_dir)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

        self.l1_5_config = self.base_config.get("layer1_5_optimization")
        if not self.l1_5_config or not self.l1_5_config.get("enabled", False):
            raise ValueError("YAML 中未啟用 layer1_5_optimization，管線中止。")

        # 自動從 system_resources 解析多進程/並行參數吃滿 CPU
        sys_res = self.base_config.get("system_resources", {})
        self.n_jobs = sys_res.get("l1_5_optuna_workers", os.cpu_count() or 4)
        max_cpu_threads = sys_res.get("max_cpu_threads", os.cpu_count() or 4)

        # 動態設定 Polars 與底層線程池，避免多進程搶核爭用
        per_worker_threads = max(1, max_cpu_threads // self.n_jobs)
        os.environ["POLARS_MAX_THREADS"] = str(per_worker_threads)
        os.environ["L1_5_OPTUNA_WORKERS"] = str(self.n_jobs)

        logger.info(f"[*] 成功加載系統資源配置: Optuna 並行 Workers = {self.n_jobs}, 每工單 Polars 線程 = {per_worker_threads}")

    def run(self, target_wl_id: str = None):
        world_lines = self.l1_5_config.get("world_lines", [])
        search_budget = self.l1_5_config.get("search_budget", {}).get("trials", 1000)

        if target_wl_id:
            matched = [wl for wl in world_lines if wl["id"].upper() == target_wl_id.upper()]
            if matched:
                world_lines = matched
            else:
                logger.info(f"[*] YAML 中未定製世界線 '{target_wl_id}'，自動開闢 SOTA 專屬世界線...")
                world_lines = [{
                    "id": target_wl_id,
                    "searcher": target_wl_id,
                    "proxy": "Qlib_Direct_RankIC",
                    "fitness": "Direct_RankIC_IR"
                }]

        if not world_lines:
            logger.error(f"找不到指定的 World Line 或未定義！")
            return

        wl = world_lines[0]
        wl_id = wl["id"]

        logger.info(f"========== 啟動尺度 {self.target_scale} 高原優化 ==========")
        logger.info(f">>> 啟動世界線 {wl_id} ...")

        searcher = ParameterSearcher(
            base_config=self.base_config,
            target_scale=self.target_scale,
            split_dir=str(self.split_dir),
            search_budget=search_budget,
            n_jobs=self.n_jobs
        )

        try:
            top_3 = searcher.run_search(wl)
        except Exception as e:
            logger.error(f"探索過程發生異常: {e}", exc_info=True)
            return

        if not top_3:
            logger.warning(f"無法找出任何有效參數。")
            return

        tournament = DeathmatchTournament(
            base_config=self.base_config,
            target_scale=self.target_scale,
            split_dir=str(self.split_dir),
            output_dir=str(self.output_path.parent)
        )

        tournament.evaluate_and_save(top_3, wl_id, self.output_path)
        logger.info("========== Layer 1.5 參數高原優化所 任務圓滿結束 ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target-scale", type=str, required=True, help="必須指定目標尺度，例如 1m, 5m 等")
    parser.add_argument("--world-line", type=str, default="CMAES_Ridge_IC")
    parser.add_argument("--split-dir", type=str, default="infra/feature_store/splits")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    split_dir = Path(args.split_dir).resolve()

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_dir = Path("configs/optimized_assets").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = auto_generate_output_name(config_path, args.target_scale, args.world_line, out_dir)
        logger.info(f"[*] 自動推導輸出檔名: {out_path.name}")

    pipeline = OptimizationPipeline(
        config_path=str(config_path),
        output_path=str(out_path),
        target_scale=args.target_scale,
        split_dir=str(split_dir)
    )
    pipeline.run(target_wl_id=args.world_line)
