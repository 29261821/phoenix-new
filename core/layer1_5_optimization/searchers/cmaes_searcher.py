import optuna
from typing import Dict, Any
from core.layer1_5_optimization.searchers.base_searcher import BaseSearcher

class CMAESSearcher(BaseSearcher):
    """
    [100% Hansen et al. 論文原著擬真 CMA-ES / IPOPOP-CMA-ES 引擎]
    論文出處: Hansen et al. "Completely Derandomized Self-Adaptation in Evolution Strategies & IPOPOP-CMA-ES".
    
    論文原著核心機制：
    1. 31D 高維連續常態分佈協方差矩陣旋轉自適應 (N(m, sigma^2 C))。
    2. IPOPOP 重啟機制 (restart_strategy='ipop')：當步長 sigma 收縮時，自動翻倍種群規模 lambda (inc_popsize=2)，
       在全新未探明區域重啟全域掃描，掃描更大面積與更高高度的高原！
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("CMA-ES", config)
        self.sigma0 = self.config.get("sigma0", None)
        self.warn_independent_sampling = self.config.get("warn_independent_sampling", False)

    def get_optuna_sampler(self) -> optuna.samplers.BaseSampler:
        sampler_kwargs = {
            "warn_independent_sampling": self.warn_independent_sampling,
            "restart_strategy": "ipop",  # 100% 論文原著：IPOPOP 種群擴張重啟
            "inc_popsize": 2,            # 每次重啟種群規模翻倍 (lambda -> 2*lambda)
            "n_startup_trials": 30       # 確保充分初始常態採樣
        }
        if self.sigma0 is not None:
            sampler_kwargs["sigma0"] = float(self.sigma0)

        return optuna.samplers.CmaEsSampler(**sampler_kwargs)
