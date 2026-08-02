#!/bin/bash
# ==============================================================================
# 這是 Zhou 專屬的自動化執行腳本
# 用途: 一鍵觸發 Layer 1 的台股數據特徵管線
# ==============================================================================

# 切換到專案根目錄，確保相對路徑正確
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit

echo "[Zhou's Pipeline] 開始執行 Layer 1 台股特徵建構管線..."

# 1. 抓取最新原始資料
echo "[Zhou's Pipeline] 步驟 1: 下載/更新原始資料..."
python3 zhou_data_engineering/auto_data_lake.py

# 2. 呼叫共用的核心模組，並指定我們要跑的台股標的
# (這會自動讀取 configs/assets/TW_STOCKS.yaml 裡的設定，並載入 zhou_data_engineering/market_data_lake/ 的原始數據)
echo "[Zhou's Pipeline] 步驟 2: 建立多尺度資料湖與特徵管線..."
python3 core/layer1_data/data_lake_builder.py --assets 2330 2308 2454

echo "[Zhou's Pipeline] 執行完畢！特徵檔已輸出至 infra/data_lake/parquet_store/TWSE/"
