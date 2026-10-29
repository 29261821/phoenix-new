# Phoenix 思想鋼印 - WSL2 跨電腦環境快速建置與複製指南

本指南用於在新電腦上快速安裝與部署 100% 相同的 Phoenix WSL2 Linux 量化尋優環境。

---

## 🛠️ 第一階段：Windows 啟用 WSL2 (僅新電腦需執行一次)

在 **Windows PowerShell (系統管理員權限)** 中執行：

```powershell
# 1. 自動安裝 WSL2 與 Ubuntu 24.04 LTS
wsl --install -d Ubuntu-24.04

# 2. 安裝完成後重啟電腦
shutdown /r /t 0
```
*重啟後打開 Ubuntu，設定新的使用者名稱與密碼。*

---

## 🚀 第二階段：執行全自動 1-Click 環境安裝

開啟 WSL2 Ubuntu 終端機，執行以下指令：

```bash
# 1. 進入本安裝目錄 (請依實際 Windows 槽位替換)
cd /mnt/c/Users/User/Desktop/專案/程式交易/RNA/思想鋼印/新版藍圖/WSL_Environment_Setup

# 2. 賦予腳本執行權限並啟動自動化安裝
chmod +x setup_wsl_env.sh
./setup_wsl_env.sh
```

此腳本將會自動完成：
- 下載並設定 Miniconda3 (Linux x86_64)
- 建立 `py311` (Python 3.11) 虛擬環境
- 一鍵安裝 **Rust-Polars**、**PyTorch**、**GPyTorch**、**HEBO 官方庫**、**Optuna**、**LightGBM**、**CatBoost** 等全套件。

---

## ⚡ 第三階段：移至 Linux 原生目錄（獲得 5x 極速 Rust/Polars I/O）

為了徹底消除 Windows /mnt/c/ 磁碟跨 OS 轉換的效能損耗，**強烈建議將專案複製到 Linux 原生主目錄 `~/phoenix`**：

```bash
# 1. 將程式碼專案複製到 Linux 原生 home 目錄
cp -r /mnt/c/Users/User/Desktop/專案/程式交易/RNA/思想鋼印/新版藍圖/phoenix ~/phoenix

# 2. 切換至 Linux 原生專案目錄
cd ~/phoenix

# 3. 啟用 py311 環境
conda activate py311
```

---

## 🧪 第四階段：環境完整性與 4 大 SOTA 選手運轉驗證

在專案目錄下執行以下單元測試指令：

```bash
PYTHONPATH=. python scratch/verify_all_searchers.py
```

當看到以下輸出時，即代表環境 100% 部署成功：
```text
  [✓] TPE 引擎 100% 通過運轉檢驗！
  [✓] HEBO 官方原生引擎 (from hebo.optimizers.hebo import HEBO) 100% 通過運轉檢驗！
  [✓] TuRBO 銷毀重啟引擎 (Uber AI Trust Region Re-spawn) 100% 通過運轉檢驗！
  [✓] CMA-ES 引擎 (IPOPOP 種群翻倍重啟) 100% 通過運轉檢驗！
```

---

## 🏁 第五階段：開始平行尋優與成果傳回

您可以在新電腦發動尋優：
```bash
python -m core.layer1_5_optimization.pipeline --config configs/assets/BTC_USDT.yaml --target-scale 1d --world-line TuRBO
```

**成果傳回**：
尋優結束後，**只需將產出的 YAML 策略檔**（例如 `configs/optimized_assets/BTC_USDT_TuRBO_1d.yaml`）傳回給主電腦即可！主電腦即可共享並接力運算！
