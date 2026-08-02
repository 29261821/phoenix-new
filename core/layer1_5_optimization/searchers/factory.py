from typing import Dict, Any
from core.layer1_5_optimization.searchers.base_searcher import BaseSearcher
from core.layer1_5_optimization.searchers.cmaes_searcher import CMAESSearcher
from core.layer1_5_optimization.searchers.tpe_searcher import TPESearcher
from core.layer1_5_optimization.searchers.hebo_searcher import HEBOSearcher
from core.layer1_5_optimization.searchers.turbo_searcher import TuRBOSearcher

def create_searcher(searcher_name: str, config: Dict[str, Any] = None) -> BaseSearcher:
    """
    Searcher 工廠模式：根據 searcher_name 動態實例化 SOTA 尋優引擎。
    支援引擎:
      - CMA-ES: 協方差矩陣自適應演化策略 (Covariance Matrix Adaptation Evolution Strategy)
      - HEBO  : 異方差演化貝氏優化 (Heteroscedastic Evolutionary Bayesian Optimization)
      - TuRBO : 信賴域貝氏優化 (Trust Region Bayesian Optimization)
      - TPE   : 樹狀 Parzen 估計器 (Tree-structured Parzen Estimator)
    """
    sname_upper = str(searcher_name).upper().strip()
    if "CMA" in sname_upper:
        return CMAESSearcher(config)
    elif "HEBO" in sname_upper:
        return HEBOSearcher(config)
    elif "TURBO" in sname_upper or "TURBO" in sname_upper:
        return TuRBOSearcher(config)
    elif "TPE" in sname_upper:
        return TPESearcher(config)
    else:
        # 預設彈性降級為 TPE 高維多變量引擎
        return TPESearcher(config)
