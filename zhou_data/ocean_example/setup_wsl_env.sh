#!/usr/bin/env bash
# =================================================================
#  Phoenix 思想鋼印 - WSL2 全套件一鍵自動化環境安裝腳本
# =================================================================

set -e

echo "================================================================="
echo "  🚀 啟動 Phoenix WSL2 Linux 全套件環境建置..."
echo "================================================================="

# 1. 檢查與安裝 Miniconda (若未安裝)
if ! command -v conda &> /dev/null; then
    echo "[*] 未檢測到 Conda，正在下載並自動安裝 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    echo "[v] Miniconda 安裝完畢！"
else
    echo "[v] 檢測到既有 Conda 環境。"
fi

# 載入 conda 指令
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# 2. 建立專屬 py311 Conda 環境
ENV_NAME="py311"
if conda env list | grep -q "$ENV_NAME"; then
    echo "[*] Conda 環境 '$ENV_NAME' 已存在，直接啟用..."
else
    echo "[*] 正在建立 Python 3.11 Conda 環境 '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.11 -y
fi

conda activate $ENV_NAME
echo "[v] 已成功切換至 Python 3.11 環境！"

# 3. 升級 pip
python -m pip install --upgrade pip setuptools wheel

# 4. 安裝需求套件 (包含 Rust-Polars, PyTorch, GPyTorch, HEBO, Optuna)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "[*] 正在從 requirements.txt 一鍵安裝全套件..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "[*] 正在手動安裝全套件..."
    pip install polars numpy scipy pandas scikit-learn lightgbm catboost optuna pyyaml torch gpytorch pymoo hebo
fi

echo "================================================================="
echo "  🎉 Phoenix WSL2 全套件環境建置成功！"
echo "  請執行: conda activate py311"
echo "================================================================="
