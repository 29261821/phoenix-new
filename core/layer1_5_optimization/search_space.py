import yaml
import logging
import copy
from typing import Dict, Any, Tuple
from core.system.logger import setup_phoenix_logger

logger = setup_phoenix_logger("L1.5_Optimization", "GLOBAL", "ALL")

class SearchSpaceParser:
    """
    [Layer 1.5 - Phase 3 Helper] 純淨簡潔搜尋空間解析器
    架構設計：
    1. 零人工塗抹 (Zero Landscape Distortion)：
       不對抽出的參數進行強行扭曲或剪裁，保持適應度地形 (Fitness Landscape) 100% 自然平滑。
    2. 純淨靜態分佈 (100% Static Distribution)：
       每一個參數使用 100% 固定的 [low, high] 靜態分佈，完全相容 CMA-ES 與 TPE 矩陣優化器。
    3. 天花板自適應 (Warmup Bars Upper Limit)：
       所有 K 線長度參數的探索上限由 YAML 的 warmup_bars 自動動態鎖定。
    """
    def __init__(self, config_dict: dict, target_scale: str):
        self.config = copy.deepcopy(config_dict)
        self.target_scale = target_scale

        # 提取該尺度的所有指標預設參數
        self.base_params = self.config.get("indicator_default_params", {}).get(target_scale, {})
        if not self.base_params:
            raise ValueError(f"YAML 缺少 {target_scale} 的 indicator_default_params 宣告！")

        # 讀取物理極限 warmup_bars 作為天花板
        self.warmup_bars = self.config.get("phase1_window", {}).get("warmup_bars", 300)
        logger.info(f"[*] Layer 1.5 探索空間鎖定物理天花板 warmup_bars = {self.warmup_bars}")

        self.flat_keys = []
        self.flat_types = []
        self.flat_defaults = []

        self._flatten()

    def _flatten(self):
        """展平指標參數字典"""
        for ind_name, params in self.base_params.items():
            if not isinstance(params, dict):
                continue
            for param_key, param_val in params.items():
                if param_key == "has_fast_slow":
                    continue
                if isinstance(param_val, (int, float)) and not isinstance(param_val, bool):
                    flat_key = f"{ind_name}.{param_key}"
                    self.flat_keys.append(flat_key)
                    self.flat_types.append(type(param_val))
                    self.flat_defaults.append(param_val)

        logger.info(f"[*] 成功展平 {self.target_scale} 參數空間，共計 {len(self.flat_keys)} 維。")

    def suggest_optuna(self, trial) -> Dict[str, Any]:
        """
        Optuna 採樣器：使用 100% 固定 [low, high] 靜態邊界，嚴格執行整數與浮點型態區隔。
        """
        suggested = {}
        lookback_keywords = ["len", "period", "_p", "lookback", "window"]

        for key, val_type, default_val in zip(self.flat_keys, self.flat_types, self.flat_defaults):
            is_lookback = any(kw in key.lower() for kw in lookback_keywords)
            if val_type == int:
                low = max(1, int(default_val * 0.1))
                high = self.warmup_bars if is_lookback else max(low + 1, int(default_val * 10.0))
                if high <= low:
                    high = low + 1
                # 嚴格使用 suggest_int，絕不使用 suggest_float 採樣整數指標！
                suggested[key] = trial.suggest_int(key, int(low), int(high))
            elif val_type == float:
                low = float(default_val * 0.1)
                high = float(default_val * 10.0)
                if high <= low:
                    high = low + 0.1
                suggested[key] = trial.suggest_float(key, low, high)

        return suggested

    def unflatten(self, flat_params: Dict[str, Any]) -> dict:
        """純淨還原 flat dict 到 YAML 巢狀字典結構 (嚴格執行型態對齊，避免型態混淆)"""
        new_params = copy.deepcopy(self.base_params)

        for flat_key, new_val in flat_params.items():
            if "." not in flat_key:
                continue
            indicator_name, param_key = flat_key.split(".", 1)
            if indicator_name in new_params and param_key in new_params[indicator_name]:
                orig_type = type(new_params[indicator_name][param_key])
                if orig_type == int:
                    new_params[indicator_name][param_key] = int(round(float(new_val)))
                elif orig_type == float:
                    new_params[indicator_name][param_key] = float(new_val)
                else:
                    new_params[indicator_name][param_key] = orig_type(new_val)

        return new_params

    def get_bounds(self) -> Dict[str, Tuple[float, float, bool]]:
        """回傳各參數之 (low, high, is_int) 字典，供原生 HEBO DesignSpace 解析"""
        bounds = {}
        lookback_keywords = ["len", "period", "_p", "lookback", "window"]
        for key, val_type, default_val in zip(self.flat_keys, self.flat_types, self.flat_defaults):
            is_lookback = any(kw in key.lower() for kw in lookback_keywords)
            if val_type == int:
                low = float(max(1, int(default_val * 0.1)))
                high = float(self.warmup_bars if is_lookback else max(low + 1, int(default_val * 10.0)))
                if high <= low:
                    high = low + 1.0
                bounds[key] = (low, high, True)
            else:
                low = float(default_val * 0.1)
                high = float(default_val * 10.0)
                if high <= low:
                    high = low + 0.1
                bounds[key] = (low, high, False)
        return bounds

    def jitter_params(self, flat_params: Dict[str, Any], jitter_ratio: float = 0.1) -> Dict[str, Any]:
        """SAM 高原鄰域擾動器"""
        import random
        jittered = copy.deepcopy(flat_params)

        for key, val in jittered.items():
            if key not in self.flat_keys:
                continue
            idx = self.flat_keys.index(key)
            val_type = self.flat_types[idx]

            multiplier = 1.0 + random.uniform(-jitter_ratio, jitter_ratio)

            if val_type == int:
                new_val = int(round(val * multiplier))
                if new_val == val:
                    new_val += random.choice([-1, 1])
                new_val = max(1, min(self.warmup_bars, new_val))
                jittered[key] = new_val
            elif val_type == float:
                jittered[key] = val * multiplier

        return jittered
