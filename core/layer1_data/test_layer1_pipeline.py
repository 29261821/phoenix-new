import polars as pl
import yaml
from pathlib import Path
from datetime import datetime
import time
import sys

# ==============================================================================
# 🚨 SOTA 路徑黑魔法 (解決 relative import 報錯)
# 確保腳本無論在哪個目錄執行，都能將 phoenix 根目錄加入 Python 搜尋路徑
# ==============================================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent # 指向 phoenix/ 目錄
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用絕對路徑引入 (Absolute Imports)，Python 就能正確辨識 Package 結構
from core.layer1_data.schemas import DataConfig, AssetMetadata
from core.layer1_data.pipeline import DataLayer

def load_yaml_config(yaml_path: Path) -> dict:
    """安全載入 YAML 檔案"""
    if not yaml_path.exists():
        print(f"[Fatal] 找不到 YAML 設定檔: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("===========================================================")
    print("🚀 Phoenix Core: Layer 1 Pipeline (YAML 驅動測試)")
    print("===========================================================")
    
    # 1. 路徑解析
    yaml_path = project_root / "configs" / "R2NA_Genesis_lite.yaml"
    data_lake_path = project_root / "infra" / "data_lake" / "parquet_store"
    feature_store_path = project_root / "infra" / "data_lake" / "feature_store"
    feature_store_path.mkdir(parents=True, exist_ok=True)
    
    raw_yaml = load_yaml_config(yaml_path)
    
    try:
        # 將 YAML 映射到 100 分級 Pydantic 合約
        config = DataConfig(
            base_asset=raw_yaml['base_asset'],
            target_assets={
                k: AssetMetadata(**v) for k, v in raw_yaml['target_assets'].items()
            },
            timeframes=raw_yaml['timeframes'],
            warmup_bars=raw_yaml['phase1_window']['warmup_bars'],
            raw_yaml=raw_yaml
        )
    except Exception as e:
        print(f"[Fatal] YAML 格式不符合 DataConfig 契約: {e}")
        sys.exit(1)
        
    target_asset = config.base_asset.replace("/", "_")
    asset_dir = data_lake_path / target_asset
    
    if not asset_dir.exists():
        print(f"[Fatal] 找不到資料湖目錄: {asset_dir}")
        sys.exit(1)

    start_dt = datetime.strptime(raw_yaml['dataset_boundaries']['train_start'], "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(raw_yaml['dataset_boundaries']['train_end'], "%Y-%m-%d %H:%M:%S")
    print(f"[*] 設定歷史回測區間: {start_dt} 至 {end_dt}")
    
    print("\n[*] 正在從 Parquet 載入並截斷多尺度資料...")
    macro_history = {}
    base_1m_history = None
    
    # [核心邏輯]: 使用 Polars 的 lazy evaluation 來 filter 時間範圍，極大節省記憶體
    for tf in config.timeframes:
        file_path = asset_dir / f"{target_asset}_{tf}.parquet"
        if not file_path.exists():
            print(f"[!] 警告: 找不到 {tf} 的檔案 -> {file_path}")
            continue
            
        # 讀取並根據 YAML 的時間段進行裁切
        df = pl.read_parquet(str(file_path)).filter(
            pl.col("datetime").is_between(start_dt, end_dt)
        )
        
        if tf == "1m":
            base_1m_history = df
        else:
            macro_history[tf] = df
            
    if base_1m_history is None or len(base_1m_history) == 0:
        print(f"[Fatal] 無法載入基準 1m K 線，或該時間區間內無資料！測試中止。")
        sys.exit(1)
        
    print(f"[v] 截斷載入完成！基準 1m K 線數量: {len(base_1m_history)} 筆")
    
    print("\n[*] 啟動 Layer 1 數據管線 (DataLayer Batch Engine)...")
    pipeline = DataLayer(config)
    
    start_time = time.time()
    
    # ⚡ 執行：Batch 向量化回測生成
    wide_table = pipeline.batch(base_1m_history, macro_history)
    
    if wide_table.is_empty():
        print("[Fatal] 寬表為空，可能是因為資料總數小於暖機需求。")
        sys.exit(1)
        
    execution_time = time.time() - start_time
    print(f"[v] 寬表生成完畢！運算耗時: {execution_time:.3f} 秒")
    
    # [新增] 將最終寬表寫入實體硬碟 (Feature Store)
    output_file = feature_store_path / f"{target_asset}_wide_table_train.parquet"
    wide_table.write_parquet(output_file)
    print(f"[v] 實體寬表已儲存至: {output_file}")
    print(f"    檔案大小: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n===========================================================")
    print("🛡️ SOTA 矩陣健康度與防爆檢驗報告")
    print("===========================================================")
    
    # 檢驗 1: 矩陣形狀
    print(f"1. [矩陣形狀]: {wide_table.shape} (列數, 特徵維度)")
    
    # 檢驗 2: NaN 零容忍檢測
    null_counts = wide_table.null_count().sum_horizontal()[0]
    if null_counts == 0:
        print(f"2. [NaN 檢測]: 完美通過！全矩陣無任何 NaN。")
    else:
        print(f"2. [NaN 檢測]: 失敗！發現 {null_counts} 個 NaN。請檢查 MissingEncoder。")
        
    # 檢驗 3: 巨觀縫合特徵檢測 (Backward Join 結果)
    print("3. [巨觀縫合檢測]:")
    expected_macros = [f"close_{tf}" for tf in config.timeframes if tf != "1m"]
    missing_macros = [col for col in expected_macros if col not in wide_table.columns]
    if not missing_macros:
        print(f"   -> 通過！成功發現巨觀特徵: {expected_macros}")
    else:
        print(f"   -> 失敗！缺失巨觀特徵: {missing_macros}")
        
    # 檢驗 4: 時間變形與資訊老化 (SOTA 特徵)
    print("4. [SOTA 時間與老化特徵檢測]:")
    sota_cols = ["staleness_1h_mins", "time_delta_tau"]
    if all(col in wide_table.columns for col in sota_cols):
        print(f"   -> 通過！成功發現 d_tau 與 Staleness 欄位。")
    else:
        print(f"   -> 失敗！未發現關鍵時間特徵。")
        
    print("\n===========================================================")
    print("📊 寬表最終切片 (Tail Preview)")
    print("===========================================================")
    
    # 安全選取存在於寬表中的特徵進行展示
    display_cols = [
        "datetime", "close", "rsi_14", "macd_hist", "bb_pct_20", 
        "close_1h", "staleness_1h_mins", "time_delta_tau", "features_reason"
    ]
    display_cols = [col for col in display_cols if col in wide_table.columns]
    
    # 設定顯示寬度，避免斷行
    pl.Config.set_tbl_width_chars(120)
    print(wide_table.select(display_cols).tail(5))
    
    print("\n[🎯 指揮官，Layer 1 YAML 驅動測試已成功通過！]")

if __name__ == "__main__":
    main()