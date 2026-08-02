from abc import ABC, abstractmethod
import optuna
from typing import Dict, Any

class BaseSearcher(ABC):
    """
    Layer 1.5 搜尋器基底抽象類別 (Abstract Base Class for Layer 1.5 Searchers)
    所有未來的優化引擎 (CMA-ES, HEBO, TuRBO, TPE) 皆需繼承此類別。
    """
    def __init__(self, searcher_name: str, config: Dict[str, Any] = None):
        self.searcher_name = searcher_name
        self.config = config or {}

    @abstractmethod
    def get_optuna_sampler(self) -> optuna.samplers.BaseSampler:
        """回傳配置完成的 Optuna Sampler 實例"""
        pass
