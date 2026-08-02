from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class AdjudicatorConfig(BaseModel):
    """
    Layer 1.6 特徵審判庭配置合約
    """
    enabled: bool = Field(True, description="是否啟用特徵審判庭")
    mode: str = Field("semi-auto", description="審判模式：'semi-auto' (僅標示不刪除), 'auto' (物理切除)")
    shadow_rejection: bool = Field(True, description="是否啟用影子特徵淘汰機制 (對抗雜訊)")
    correlation_threshold: float = Field(0.8, description="Spearman 相關性分群門檻 (0.0~1.0)")
    
    # LightGBM 訓練參數 (供影子與特徵重要度擷取使用)
    training_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "n_estimators": 100,
        }
    )
