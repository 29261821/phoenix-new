import polars as pl
import asyncio
import logging
from typing import Optional, AsyncGenerator
import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BinanceClient")

class BinanceClient:
    """
    [Layer 1 - ⚡ SOTA 核心] 幣安非同步與實盤串流客戶端
    職責：
    1. 提供帶有「指數退避 (Exponential Backoff)」防禦機制的歷史資料拉取 (REST)
    2. 提供高速零延遲的 WebSocket 即時 K 線串流 (Streaming)
    """
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True, # CCXT 內建的基礎速率限制
            'options': {
                'defaultType': 'future', # 預設切換為合約市場 (依需求可改為 'spot')
            }
        }
        
        # REST API 非同步客戶端 (用於歷史回補)
        self.rest_client = ccxt_async.binance(config)
        
        # WebSocket 即時串流客戶端 (用於實盤推播)
        self.ws_client = ccxt_pro.binance(config)

    async def fetch_ohlcv_with_backoff(self, symbol: str, timeframe: str, limit: int = 1000, max_retries: int = 5) -> pl.DataFrame:
        """
        [防爆核心] 獲取歷史 K 線，並具備指數退避重試機制
        """
        retries = 0
        while retries < max_retries:
            try:
                # 非同步拉取資料
                ohlcv = await self.rest_client.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    return pl.DataFrame()

                # 高速轉換為 Polars DataFrame
                df = pl.DataFrame(
                    ohlcv, 
                    schema=["timestamp", "open", "high", "low", "close", "volume"],
                    orient="row"
                ).with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))
                
                return df
                
            except ccxt_async.RateLimitExceeded as e:
                wait_time = 2 ** retries # 1, 2, 4, 8, 16 秒
                logger.warning(f"[Rate Limit] 觸發 API 限制。{wait_time} 秒後重試... ({retries+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                retries += 1
                
            except ccxt_async.NetworkError as e:
                wait_time = 2 ** retries
                logger.warning(f"[Network Error] 網路連線異常: {e}。{wait_time} 秒後重試... ({retries+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                retries += 1
                
            except Exception as e:
                logger.error(f"[Fatal Error] 未知異常: {e}")
                raise e
                
        raise Exception(f"無法獲取 {symbol} 數據：已達最大重試次數 {max_retries}。")

    async def stream_live_kline(self, symbol: str, timeframe: str) -> AsyncGenerator[pl.DataFrame, None]:
        """
        [實盤引擎] 透過 WebSocket 訂閱實時 K 線變化
        每次 K 線更新 (Tick) 或收盤時，Yield 出最新的一根 K 線特徵
        """
        logger.info(f"[*] 啟動 WebSocket 串流：{symbol} ({timeframe})")
        while True:
            try:
                # CCXT Pro 的 watch_ohlcv 會自動處理訂閱與斷線重連
                ohlcv_list = await self.ws_client.watch_ohlcv(symbol, timeframe)
                
                # 取得最新的一根 K 線 (包含尚未收盤的即時跳動)
                latest_candle = ohlcv_list[-1]
                
                df = pl.DataFrame(
                    [latest_candle],
                    schema=["timestamp", "open", "high", "low", "close", "volume"],
                    orient="row"
                ).with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))
                
                yield df
                
            except Exception as e:
                logger.error(f"[WebSocket Error] 串流中斷: {e}。5 秒後嘗試重新建立連線...")
                await asyncio.sleep(5)

    async def close_connections(self):
        """安全關閉所有連線"""
        logger.info("[*] 關閉 Binance 連線...")
        await self.rest_client.close()
        await self.ws_client.close()

# === 模組測試區 (使用 asyncio 運行) ===
async def _test_runner():
    # 建立客戶端 (公開資料無需 API Key)
    client = BinanceClient()
    symbol = "BTC/USDT"
    
    print("\n--- 測試 1: REST API (帶有防禦機制) ---")
    df = await client.fetch_ohlcv_with_backoff(symbol, "1m", limit=5)
    print(df)
    
    print("\n--- 測試 2: WebSocket 實盤串流 (監聽 3 次 Tick 後退出) ---")
    count = 0
    async for live_df in client.stream_live_kline(symbol, "1m"):
        print(f"[Tick {count+1}] 即時價格: {live_df['close'][0]}, 累積成交量: {live_df['volume'][0]}")
        count += 1
        if count >= 3:
            break
            
    await client.close_connections()

if __name__ == "__main__":
    # 在 Windows 環境下避免 RuntimeError
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(_test_runner())