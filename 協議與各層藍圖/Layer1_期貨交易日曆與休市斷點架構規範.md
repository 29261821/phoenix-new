# Layer 1 期貨交易日曆與休市斷點架構規範 (100% Futures Target Specification)

此文件為 Phoenix 體系針對 **100% 傳統期貨落地 (CME MBT, MNQ, NQ, ES, CL, GC)** 之交易日曆、休市斷點與節假日 Mask 權規藍圖。

---

## Ⅰ. 期貨與加密貨幣之底層物理差異

在 Phoenix 體系中，加密貨幣 (Crypto) 僅作為開發與實驗過程之便利測試標的；**最終 100% 落地對象為傳統交易所期貨 (CME / Eurex / HKEX)**。

傳統期貨與 Crypto 的核心物理差異如下：
1. **非 24/7 連續交易**：期貨每日存在結算休市 (如 CME 每日 17:00 ~ 18:00 ET 暫停交易)。
2. **週末物理斷點 (Weekend Closures)**：週五 17:00 ET 關盤至週日 18:00 ET 開盤。
3. **國定假日 (Holidays)**：感恩節 (Thanksgiving)、聖誕節 (Christmas)、元旦 (New Year's Day) 全天休市或提早收盤。
4. **本專案邊界宣告**：**暫不考慮 Tick 級微秒訂單簿引擎**，全身心聚焦於 Bar 級別 (1m, 5m, 15m, 1h, 1d) 數據流之正確性與防過擬合。

---

## Ⅱ. 假性低波動度污染 (Fake Low-Volatility Distortion) 防線

若直接將 Crypto 的 Forward Fill (向後補值) 套用至期貨：
- **災難**：在週五收盤至週日開盤的 48 小時內，若補滿平坦 K 線，技術指標（如 `AVO`, `ADX`, `ATR`, `Kalman_Filter`）會誤判市場為「波動率極低且無風險」，從而發出錯誤的重倉買入訊號！
- **解決方案**：
  1. 引入 `FuturesMarketCalendar` 處理器，在物理 DataLake 與 FeatureStore 層級剔除休市/假日 Bars。
  2. 標記 `is_session_start` (開盤第一根 K 線)，讓滾動視窗跳過跨休市區間的虛假變異數。

---

## Ⅲ. YAML 配置宣告規範 (`configs/assets/CME_MNQ.yaml`)

```yaml
asset: "MNQ"
exchange: "CME"
market_type: "FUTURES"

# 期貨交易日曆與休市斷點契約
market_calendar:
  timezone: "America/New_York"
  session_breaks:
    - start: "17:00"
      end: "18:00"
  weekend_closure:
    start_day: "Friday"
    start_time: "17:00"
    end_day: "Sunday"
    end_time: "18:00"
  holidays:
    - "2026-01-01" # New Year's Day
    - "2026-01-19" # Martin Luther King Jr. Day
    - "2026-02-16" # Washington's Birthday
    - "2026-04-03" # Good Friday
    - "2026-05-25" # Memorial Day
    - "2026-06-19" # Juneteenth
    - "2026-07-03" # Independence Day
    - "2026-09-07" # Labor Day
    - "2026-11-26" # Thanksgiving Day
    - "2026-12-25" # Christmas Day

trading_economics:
  margin: 2200.0           # CME MNQ 每口初始保證金 ($)
  contract_size: 2.0       # 合約乘數 ($2 x Index Point)
  tick_size: 0.25          # 最小跳動點
  tick_value: 0.50         # 最小跳動值 ($)
  slippage_ticks: 1.0      # 滑價 (1 Tick = $0.50)
  spread_ticks: 1.0        # 買賣價差
  commission_per_order: 0.62 # CME 雙邊單腳手續費 ($)
  initial_capital: 20000.0 # 回測初始資金 ($)
```

---

## Ⅳ. 鋼印更新：期貨交易日曆防線 (Steel Stamp Rule 12)

已將此規範作為**第 12 條思想鋼印**，正式納入 Phoenix 體系：

### 12. 期貨交易日曆與休市斷點防線 (Futures Calendar & Session Break Defense)
- **原則**：當處理 100% 落地之期貨數據時，必須嚴格加載 `market_calendar`，濾除每日 17:00-18:00 結算休市、週末休市與國定假日。
- **紅線**：**嚴禁在休市與假日區間進行 Forward Fill 補線**，防止指標產生假性低波動度 (Zero Volatility) 訊號扭曲！
