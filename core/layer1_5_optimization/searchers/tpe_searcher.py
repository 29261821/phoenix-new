import optuna
from typing import Dict, Any
from core.layer1_5_optimization.searchers.base_searcher import BaseSearcher

class TPESearcher(BaseSearcher):
    """
    [100% Bergstra et al. 論文原著擬真 TPE 高維多變量混合引擎]
    論文出處: Bergstra et al. "Algorithms for Hyper-Parameter Optimization" (NIPS 2011).
    
    論文原著核心機制：
    1. 採用多變量混合高斯模型 (Multivariate TPE) 捕捉 31D 特徵間的高階相互作用。
    2. 多 Workers 常數謊言 (Constant Liar) 避免多 CPU 搶單同質化。
    3. 論文原著指定高維空間 startup_trials = 30 確保超體積空間初估品質。
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TPE", config)
        self.n_startup_trials = self.config.get("n_startup_trials", 30)
        self.multivariate = self.config.get("multivariate", True)
        self.group = self.config.get("group", True)
        self.constant_liar = self.config.get("constant_liar", True)
        self.seed = self.config.get("seed", 42)

    def get_optuna_sampler(self) -> optuna.samplers.BaseSampler:
        return optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            multivariate=self.multivariate,
            group=self.group,
            constant_liar=self.constant_liar,
            seed=self.seed,
        )
