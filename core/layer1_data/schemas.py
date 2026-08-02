import polars as pl
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Any, Union

class AssetMetadata(BaseModel):
    """
    [Layer 1] 資產物理與制度描述 (Asset Semantics)
    提供給下游模型學習不同資產的物理特性 (如時區差異、跳動單位)
    """
    asset_id: str = Field(..., description="標的代碼 (e.g., 'BTC/USDT', 'TX00')")
    asset_type: Literal["crypto", "equity_index", "commodity", "forex"] = Field(
        ..., description="資產大類 (決定了模型對待其休市與假日的態度)"
    )
    timezone: str = Field(..., description="交易所時區 (e.g., 'UTC', 'America/New_York')")
    tick_size: float = Field(..., gt=0, description="最小跳動單位 (必須大於 0)")

class AssetDataConfig(BaseModel):
    """
    [Layer 1] 單一商品超參數配置契約 (MCP 級解耦)
    定義該商品專屬的物理特徵、多寬表尺度與出戰特徵。
    """
    base_asset: str = Field(..., description="基準資產 (e.g., 'BTC/USDT')")
    target_assets: Dict[str, AssetMetadata] = Field(..., description="全市場標的註冊表 (包含基準資產與其他需要跨商品關聯的資產)")
    
    base_timeframe: str = Field(
        default="1m",
        description="本商品在資料湖中最底層的物理 K 線尺度 (e.g. '1m', '5m')"
    )
    macro_timeframes: List[str] = Field(
        default=["5m", "15m", "1h", "1d"], 
        description="需進行 Backward Join 縫合的巨觀時間框架"
    )
    
    # 🚨 [SOTA 升級] 物理暖機與動態 YAML 路由合約
    warmup_bars: int = Field(
        default=300, 
        description="系統啟動或回測時的強制暖機 K 線數量，防範 EMA/MACD 等指標初始發散"
    )
    raw_yaml: Dict[str, Any] = Field(
        default_factory=dict, 
        description="解析後的完整 YAML 配置字典，供 FeatureProxy 進行動態特徵註冊與參數覆寫"
    )

    @field_validator("target_assets")
    @classmethod
    def check_base_asset_in_targets(cls, target_assets: Dict[str, AssetMetadata], info: Any) -> Dict[str, AssetMetadata]:
        """防呆驗證：基準資產必須存在於註冊表中"""
        # 兼容 Pydantic v2 的 ValidationInfo
        base_asset = info.data.get("base_asset") if hasattr(info, "data") else None
        if base_asset and base_asset not in target_assets:
            raise ValueError(f"基準資產 {base_asset} 未在 target_assets 註冊表中定義！")
        return target_assets

class MarketObservation(BaseModel):
    """
    [Layer 1] 核心輸出契約 (傳遞給 Layer 2 的標準介面)
    確保進入神經網路的數據具備完美的物理封裝與防爆遮罩。
    """
    timestamp: int = Field(..., description="全局基準 Heartbeat (Unix Timestamp 秒/毫秒)")
    asset_id: str = Field(..., description="資料所屬標的")
    
    # --- 三維遺失編碼 (Missingness Encoding) ---
    features_value: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="實際數值或 ffill 後的連續數值矩陣 (若為深度學習模式，則為各尺度獨立之字典)")
    features_mask: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="Boolean: True 表示該 T 刻有真實交易，False 為補值")
    features_reason: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="IntEnum: 0=正常, 1=休市, 2=熔斷, 3=API斷線")
    
    # --- 時間語意特徵 (Event Clock & Time Deformation) ---
    time_features: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="包含 time_delta_tau, minutes_since_open 等")

    class Config:
        # 允許 Pydantic 裝載 Polars DataFrame 這種非標準 Python 類型
        arbitrary_types_allowed = True

    @field_validator("features_mask", "features_reason", "time_features")
    @classmethod
    def check_dataframe_length_match(cls, df: pl.DataFrame, info: Any) -> pl.DataFrame:
        """矩陣形狀防呆驗證：所有特徵維度的長度必須嚴格對齊"""
        if not hasattr(info, "data"): return df
        
        # 取得 features_value 作為長度基準
        base_df = info.data.get("features_value")
        if isinstance(base_df, pl.DataFrame) and len(df) != len(base_df):
            field_name = info.field_name if hasattr(info, "field_name") else "unknown"
            raise ValueError(f"維度對齊錯誤！{field_name} 的長度 ({len(df)}) 與 features_value ({len(base_df)}) 不一致。")
        return df