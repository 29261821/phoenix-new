# README for Layer 1.5 Optimization

## 快慢線指標標註規則（Fast/Slow Indicator Annotation Rules）

### 目的
- **確保第 1.5 層搜尋空間能正確辨識快慢線指標**，以產生 `fast_len`、`slow_len` 參數。
- 防止因 YAML 中缺少 `has_fast_slow: true` 而導致搜尋空間錯誤或程式在執行時中止。

### 規則概述
1. **程式碼層面**
   - 任何位於 `core/layer1_data/features_library/Indicators/` 的 Python 檔案，只要同時出現 `fast_period` 與 `slow_period`（不區分大小寫），即視為 **快慢線指標**。
   - 目前已自動偵測的快慢線指標包括 `MACD`, `N_MACD`, `Rolling_OBV_MACD`, `AO`, `Geography`, `Trend_Health` 等。
2. **YAML 標註層面**
   - 在每個資產的 YAML（`phoenix/configs/assets/*.yaml`）的 `indicator_default_params` 區塊，對於快慢線指標必須加入布林欄位 `has_fast_slow: true`。
   - 範例（`BTC_USDT.yaml`）
   ```yaml
   indicator_default_params:
     "1m":
       MACD: { source: "close", fast_period: 12, slow_period: 26, signal_period: 9, has_fast_slow: true }
   ```
   - 若指標不屬於快慢線，**不需要**此欄位，若不小心加入會被視為已標註，系統不會阻止（但建議保持一致性）。

### 驗證與自動補全工具
- **模組** `core/layer1_5_optimization/fast_slow_validator.py`
  - `FastSlowValidator.validate_all()`：返回 `(all_ok, missing)`，其中 `missing` 為 `[(yaml_path, indicator_name), ...]`。
  - `FastSlowValidator.fix_all()`：自動在缺失的 YAML 中加入 `has_fast_slow: true`，同時在同目錄生成 `*.bak` 備份檔。
- **CLI** `scripts/validate_fast_slow_yaml.py`
  - `--check` 只檢查並列出缺失項目。
  - `--fix` 自動補全（會備份原檔）。
  - 使用方式（在專案根目錄）:
    ```bash
    python -m scripts.validate_fast_slow_yaml --check   # 檢查
    python -m scripts.validate_fast_slow_yaml --fix    # 補全
    ```

### 第 1.5 層自動驗證
- `core/layer1_5_optimization/search_space.py` 已在 `SearchSpaceParser.__init__` 中呼叫 `FastSlowValidator.validate_all()`。
- 若發現缺失，會 **拋出 `ValueError`**，訊息會列出所有缺少 `has_fast_slow` 的檔案與指標，提醒使用者先執行 `--fix` 再重新跑搜尋。

### 常見問題與排除方法
| 症狀 | 可能原因 | 解決方式 |
|------|----------|----------|
| 程式在 `SearchSpaceParser` 初始化時拋出 `ValueError`，列出缺失檔案 | 某些 YAML 未標註 `has_fast_slow` | 執行 `python -m scripts.validate_fast_slow_yaml --fix`，或手動在該 YAML 中加入 `has_fast_slow: true`。
| 新增自訂指標（例如 `MyCustom.py`）但未被偵測為快慢線 | 該指標沒有 `fast_period`/`slow_period`，或名稱使用不同關鍵字。 | 若此指標實際需要快慢線，請在程式碼中使用 `fast_period` 與 `slow_period`；若不需要快慢線，則不必在 YAML 加入標註。
| `FastSlowValidator.fix_all()` 沒有修改預期檔案 | 可能因指標名稱拼寫不一致，或 YAML 結構不同（例如使用了別名） | 確認指標名稱在 YAML 與 `Indicators` 資料夾中完全相同（不含副檔名），必要時手動修正。

### 建議工作流程
1. **開發新指標** → 確認程式碼中使用 `fast_period`、`slow_period`（若適用）。
2. **新增資產 YAML** → 為相應指標加入 `has_fast_slow: true`。
3. **執行測試** → `python -m scripts.validate_fast_slow_yaml --check` 確認無缺失。
4. **運行第 1.5 層搜尋** → 若所有檢查通過，`SearchSpaceParser` 正常運作。

---
*此 README 為第 1.5 層的官方操作說明，請在變更指標或 YAML 時遵循上述規則，以確保系統智慧化判斷不會因遺漏標註而中斷。*
