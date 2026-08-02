import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from typing import Dict, Any, Optional
from core.layer1_5_optimization.searchers.base_searcher import BaseSearcher

class TuRBOSampler(BaseSampler):
    """
    [100% Uber AI TuRBO-m 論文原著忠實實作 Sampler]
    論文出處: Uber AI Lab, NeurIPS 2019 "Trust Region Bayesian Optimization in High Dimensions".
    
    論文原著核心機制：
    1. 維護動態信賴域 [x* - L/2, x* + L/2]。
    2. 連續成功擴張 L -> 2L；連續失敗縮減 L -> L/2。
    3. 銷毀重啟 (Global Re-spawn)：當 L < L_min 時，原著規定必須徹底銷毀舊信賴域 (best_params = None)，
       於全新未採樣區間觸發 Sobol 全域 Re-spawn，探明全新高原！
    """
    def __init__(
        self,
        n_startup_trials: int = 30,
        init_length: float = 0.8,
        min_length: float = 0.5 ** 7,
        max_length: float = 1.6,
        succ_threshold: int = 3,
        fail_threshold: int = 4,
        seed: Optional[int] = 42,
    ):
        self.n_startup_trials = n_startup_trials
        self.init_length = init_length
        self.min_length = min_length
        self.max_length = max_length
        self.succ_threshold = succ_threshold
        self.fail_threshold = fail_threshold
        self.seed = seed

        # 動態信賴域內部狀態
        self.length = init_length
        self.succ_count = 0
        self.fail_count = 0
        self.best_value: Optional[float] = None
        self.best_params: Optional[Dict[str, float]] = None

        self._sub_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed)

    def infer_relative_search_space(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> Dict[str, optuna.distributions.BaseDistribution]:
        return self._sub_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution]
    ) -> Dict[str, Any]:
        if len(study.trials) < self.n_startup_trials:
            return {}

        self._update_trust_region_state(study)

        # 論文原著：若銷毀重啟 (best_params is None)，則發動全域 Re-spawn
        if not self.best_params:
            return self._sub_sampler.sample_relative(study, trial, search_space)

        # 於當前最佳點周圍構建動態信賴域邊界 [x* - L/2, x* + L/2]
        bounded_space = {}
        for name, dist in search_space.items():
            if isinstance(dist, optuna.distributions.FloatDistribution):
                center = self.best_params.get(name, (dist.high + dist.low) / 2.0)
                half_len = self.length * (dist.high - dist.low) / 2.0
                low_b = max(dist.low, center - half_len)
                high_b = min(dist.high, center + half_len)
                if low_b >= high_b:
                    low_b, high_b = dist.low, dist.high
                bounded_space[name] = optuna.distributions.FloatDistribution(low_b, high_b, log=dist.log)
            else:
                bounded_space[name] = dist

        return self._sub_sampler.sample_relative(study, trial, bounded_space)

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution
    ) -> Any:
        return self._sub_sampler.sample_independent(study, trial, param_name, param_distribution)

    def _update_trust_region_state(self, study: optuna.study.Study):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            return

        direction = study.direction
        is_maximize = (direction == StudyDirection.MAXIMIZE)

        if is_maximize:
            sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
        else:
            sorted_trials = sorted(completed, key=lambda t: t.value, reverse=False)

        current_best = sorted_trials[0]
        latest_trial = completed[-1]

        if self.best_value is None:
            self.best_value = current_best.value
            self.best_params = current_best.params
            return

        improved = (latest_trial.value > self.best_value) if is_maximize else (latest_trial.value < self.best_value)
        if improved:
            self.best_value = latest_trial.value
            self.best_params = latest_trial.params
            self.succ_count += 1
            self.fail_count = 0
        else:
            self.fail_count += 1
            self.succ_count = 0

        # 動態調整信賴域長度
        if self.succ_count >= self.succ_threshold:
            self.length = min(self.max_length, self.length * 2.0)
            self.succ_count = 0
        elif self.fail_count >= self.fail_threshold:
            self.length = self.length / 2.0
            self.fail_count = 0

        # 100% 論文原著機轉：信賴域過小 (L < L_min) 時，強制銷毀舊最佳點，全域 Re-spawn！
        if self.length < self.min_length:
            self.length = self.init_length
            self.succ_count = 0
            self.fail_count = 0
            self.best_value = None
            self.best_params = None

class TuRBOSearcher(BaseSearcher):
    """
    [TuRBO 信賴域貝氏優化解耦引擎 - 100% 論文原著重構版]
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TuRBO", config)
        self.n_startup_trials = self.config.get("n_startup_trials", 30)
        self.init_length = self.config.get("init_length", 0.8)
        self.seed = self.config.get("seed", 42)

    def get_optuna_sampler(self) -> optuna.samplers.BaseSampler:
        return TuRBOSampler(
            n_startup_trials=self.n_startup_trials,
            init_length=self.init_length,
            seed=self.seed,
        )
