import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from core.layer1_5_optimization.searchers.base_searcher import BaseSearcher

logger = logging.getLogger("L1.5_HEBO_Searcher")

class HEBOSearcher(BaseSearcher):
    """
    [100% 論文原著擬真 HEBO 異方差演化貝氏優化引擎 - 32 多核並行吃滿版]
    NeurIPS 2020 黑盒尋優大賽總冠軍原生引擎 (from hebo.optimizers.hebo import HEBO)。
    
    論文核心數學機轉與並行加速：
    1. Box-Cox / Power Transform 非線性輸入與輸出空間拉直變換。
    2. 異方差高斯過程 (Heteroscedastic GP)，動態建模空間非同方差噪聲以懲罰陡峭孤峰。
    3. EI / UCB / PI 帕累托前沿多目標 Acquisition 選點。
    4. 32-Workers 批次採集 (Batch Acquisition opt.suggest(n_suggestions=batch_size)) 
       搭配 ThreadPoolExecutor 多進程/多線程並行評估，徹底吃滿 100% CPU 算力！
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("HEBO", config)
        self.n_startup_trials = self.config.get("n_startup_trials", 30)
        self.seed = self.config.get("seed", 42)
        self.n_jobs = self.config.get("n_jobs", 32)

    def run_native_hebo_search(self, search_space_parser: Any, evaluator_fn: Any, total_trials: int = 1000, n_jobs: int = 32) -> List[Dict[str, Any]]:
        """
        原汁原味執行 100% 論文原著 HEBO 尋優流程 (32 多核並行加速)
        """
        design_space_dict = []
        param_bounds = search_space_parser.get_bounds()
        
        for name, (low, high, is_int) in param_bounds.items():
            if is_int:
                design_space_dict.append({
                    "name": name,
                    "type": "int",
                    "lb": int(low),
                    "ub": int(high)
                })
            else:
                design_space_dict.append({
                    "name": name,
                    "type": "num",
                    "lb": float(low),
                    "ub": float(high)
                })

        space = DesignSpace().parse(design_space_dict)
        opt = HEBO(space=space, rand_sample=self.n_startup_trials, model_name='gp')

        workers = max(1, n_jobs if n_jobs is not None else self.n_jobs)
        logger.info(f"[🚀] 發動 100% 論文原著 HEBO 官方引擎 (模型: GP, 預算: {total_trials} 次, 並行 Workers: {workers} 100% 吃滿 CPU)...")
        
        best_candidates = []
        batch_size = workers
        completed_trials = 0

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            while completed_trials < total_trials:
                curr_batch = min(batch_size, total_trials - completed_trials)
                try:
                    rec = opt.suggest(n_suggestions=curr_batch)
                except Exception as ex:
                    logger.warning(f"[HEBO] 採集選點微幅調整: {ex}，進行安全隨機重構...")
                    rec = space.sample(curr_batch)

                param_dicts = [rec.iloc[k].to_dict() for k in range(len(rec))]
                
                # 並行發動評估器吃滿多核 CPU
                fitness_results = list(executor.map(evaluator_fn, param_dicts))

                y_obs_list = []
                for k, raw_fitness in enumerate(fitness_results):
                    fitness_val = float(raw_fitness)
                    if fitness_val <= -990.0:
                        logger.error(
                            f"[❌ HEBO 帶入無效參數抓錯日誌] 試驗 #{completed_trials + k + 1} 失敗/產生無效值！\n"
                            f"👉 帶入之參數組合: {param_dicts[k]}\n"
                            f"👉 返回適應度: {fitness_val}"
                        )
                        y_obs_list.append([100.0 + np.random.uniform(-1e-4, 1e-4)])
                    else:
                        y_obs_list.append([-fitness_val + np.random.uniform(-1e-5, 1e-5)])
                        best_candidates.append({
                            "params": param_dicts[k],
                            "fitness": fitness_val
                        })

                opt.observe(rec, np.array(y_obs_list))
                completed_trials += len(rec)

                if completed_trials % max(1, (total_trials // 10)) == 0 or completed_trials <= batch_size:
                    current_best_score = max([c["fitness"] for c in best_candidates]) if best_candidates else -999.0
                    logger.info(f"👉 [HEBO 進度 {completed_trials/total_trials*100:5.1f}%] 已算 {completed_trials}/{total_trials} 次 | 當前最佳 IC: {current_best_score:.4f}")

        best_candidates.sort(key=lambda x: x["fitness"], reverse=True)
        return best_candidates[:3]

    def get_optuna_sampler(self):
        """相容層降級保護"""
        import optuna
        return optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=self.seed
        )
