import polars as pl
import importlib
import logging
from typing import Dict, Any, List

logger = logging.getLogger("FeatureProxy")

class FeatureProxy:
    """
    [Layer 1 - ⚡ SOTA 中樞] 動態特徵路由代理
    職責：讀取 YAML 出戰名單，動態映射並載入 features_library 中的指標。
    防爆機制：Fail-fast 預載入、Namespace 碰撞隔離、單次圖優化計算。
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self.plan = config_dict.get("meta_features_plan", [])
        self.defaults = config_dict.get("indicator_default_params", {})
        self.module_cache = {}
        
        # [Fail-Fast 紅線] 系統啟動時立刻驗證並預載入所有模組，防止實盤中 I/O 阻塞或崩潰
        self._preload_modules()

    def _preload_modules(self):
        """預先編譯並快取所需的特徵腳本"""
        logger.info(f"[*] FeatureProxy 開始預載入特徵模組 (共 {len(self.plan)} 項任務)...")
        for item in self.plan:
            indicator_name = item["indicator"]
            if indicator_name not in self.module_cache:
                try:
                    # 動態載入您規劃的特徵軍火庫路徑
                    module_path = f"core.layer1_data.features_library.Indicators.{indicator_name}"
                    mod = importlib.import_module(module_path)
                    
                    # 確保 H 接口與 G 接口存在
                    if not hasattr(mod, "calculate") or not hasattr(mod, f"adapt_{indicator_name}"):
                        raise AttributeError(f"模組 {indicator_name} 缺乏標準 H/G 接口契約。")
                        
                    self.module_cache[indicator_name] = mod
                    logger.debug(f"[+] 成功載入特徵模組: {indicator_name}")
                except ImportError as e:
                    logger.error(f"[!] 找不到特徵模組: {indicator_name}。請檢查 YAML 與檔案路徑。")
                    raise e

    def extract_features(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        [多核極限壓榨] 遍歷 YAML 出戰名單，打包所有 Lazy Expressions 並一次性執行。
        """
        exprs: List[pl.Expr] = []
        close_expr = pl.col("close")
        
        for item in self.plan:
            uid = item["id"]  # 來自 YAML 的 Unique ID (e.g., 'MACD_Fast')
            indicator_name = item["indicator"]
            mod = self.module_cache[indicator_name]
            
            # 1. 參數層疊合併：基礎預設值 -> 尺度專屬預設值 -> YAML 出戰名單覆蓋
            base_ind_defaults = self.defaults.get("1m", {}).get(indicator_name, {}).copy()
            tf_defaults = self.defaults.get(timeframe, {}).get(indicator_name, {})
            if isinstance(tf_defaults, dict):
                base_ind_defaults.update(tf_defaults)
            
            params = base_ind_defaults
            if "params" in item and isinstance(item["params"], dict):
                params.update(item["params"])
                
            # 2. 呼叫 H 接口 (人類與策略語意)
            # 傳入 df 是因為部分指標 (如 SMC, ATR) 需要 high/low 進行 Pivot 計算
            h_output = mod.calculate(df=df, params=params)
            
            # 3. 呼叫 G 接口 (深度學習寬表特徵映射)
            adapt_func = getattr(mod, f"adapt_{indicator_name}")
            g_output: Dict[str, pl.Expr] = adapt_func(h_output=h_output, close=close_expr, params=params)
            
            # 4. [防碰撞紅線] 強制重命名
            # 為了允許 YAML 呼叫多次相同的指標 (如 MACD_15m, MACD_1d)，我們必須加上 UID 前綴
            for feat_name, expr in g_output.items():
                safe_alias = f"{uid}_{feat_name}"
                exprs.append(expr.alias(safe_alias))
                
        # 5. [效能與記憶體紅線] 分批提交給 Polars 引擎
        # 若一次提交 400 個複雜的指標 Expr (包含大量的 rolling/ewm)，Polars 會在底層展開極巨大的 DAG
        # 並為每個節點並發分配巨大的記憶體池，導致「記憶體秒吃滿」甚至 OOM 崩潰。
        # 我們將 Expr 改為逐一提交 (Chunk = 1)，用極微小的時間換取絕對的記憶體安全！
        if exprs:
            for expr in exprs:
                df = df.with_columns([expr])
            
        return df