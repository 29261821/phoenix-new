import pathlib, yaml, re, importlib
from typing import List, Dict, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
INDICATOR_ROOT = PROJECT_ROOT / "phoenix" / "core" / "layer1_data" / "features_library" / "Indicators"
ASSET_YAML_ROOT = PROJECT_ROOT / "phoenix" / "configs" / "assets"

FAST_PATTERN = re.compile(r"\b(fast|fast_period|fast_len|momentum_p)\b", re.IGNORECASE)
SLOW_PATTERN = re.compile(r"\b(slow|slow_period|slow_len|inertia_p)\b", re.IGNORECASE)

def _is_fast_slow_indicator(file_path: pathlib.Path) -> bool:
    """若檔案宣告 PARAM_CONSTRAINTS (含 fast_slow_pairs/ordered_chains) 或文字包含快慢線，回傳 True。"""
    ind_name = file_path.stem
    try:
        mod = importlib.import_module(f"core.layer1_data.features_library.Indicators.{ind_name}")
        constraints = getattr(mod, "PARAM_CONSTRAINTS", {})
        if constraints and ("fast_slow_pairs" in constraints or "ordered_chains" in constraints):
            return True
    except Exception:
        pass

    try:
        content = file_path.read_text(encoding="utf-8")
        return bool(FAST_PATTERN.search(content) and SLOW_PATTERN.search(content))
    except Exception:
        return False

def _list_fast_slow_indicators() -> List[str]:
    indicators = []
    for p in INDICATOR_ROOT.iterdir():
        if p.is_file() and p.suffix == ".py" and _is_fast_slow_indicator(p):
            indicators.append(p.stem)
    return indicators

def _load_yaml(yaml_path: pathlib.Path) -> Dict:
    try:
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def _save_yaml(yaml_path: pathlib.Path, data: Dict):
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

class FastSlowValidator:
    @staticmethod
    def _list_fast_slow_indicators() -> List[str]:
        return _list_fast_slow_indicators()

    @staticmethod
    def validate_all() -> Tuple[bool, List[Tuple[pathlib.Path, str]]]:
        fast_slow_inds = set(FastSlowValidator._list_fast_slow_indicators())
        missing: List[Tuple[pathlib.Path, str]] = []
        for yaml_file in ASSET_YAML_ROOT.glob("*.yaml"):
            cfg = _load_yaml(yaml_file)
            for tf, tf_cfg in cfg.get("indicator_default_params", {}).items():
                if not isinstance(tf_cfg, dict):
                    continue
                for ind_name, ind_cfg in tf_cfg.items():
                    if ind_name in fast_slow_inds:
                        if not isinstance(ind_cfg, dict) or not ind_cfg.get("has_fast_slow"):
                            missing.append((yaml_file, ind_name))
        return (len(missing) == 0, missing)

    @staticmethod
    def fix_all() -> List[Tuple[pathlib.Path, str]]:
        ok, missing = FastSlowValidator.validate_all()
        if ok:
            return []
        for yaml_path, ind in missing:
            data = _load_yaml(yaml_path)
            for tf, tf_cfg in data.get("indicator_default_params", {}).items():
                if isinstance(tf_cfg, dict) and ind in tf_cfg:
                    if not isinstance(tf_cfg[ind], dict):
                        tf_cfg[ind] = {}
                    tf_cfg[ind]["has_fast_slow"] = True
            backup_path = yaml_path.with_suffix(".bak")
            backup_path.write_bytes(yaml_path.read_bytes())
            _save_yaml(yaml_path, data)
        return missing
