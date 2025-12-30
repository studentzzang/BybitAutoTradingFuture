import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# ====== 사용자 설정 변수 ======
SYMBOL = ["PUMPFUNUSDT"]
LEVERAGE = 5
TIMEFRAME = [3,5, 15,30]
RSI_PERIOD = [14]
EMA_PERIOD_ARR = [20,50,100]  # EMA 기간을 배열로 설정 (예: 100선, 200선 비교)
EQUITY = 100.0
START = "2025-02-01"
END = "2025-11-12"
OUT_DIR = "test"

MAX_CANDLES = 50000 

# RSI 트리거 값 (확인 진입용)
OPEN_SHORT_RSI = 70.0   
OPEN_LONG_RSI  = 30.0   

# TP / SL 설정
TP_ROE_ARR   = [6,10, 15]
SL_ROE_ARR   = [6, 10, 15]

session = HTTP()

# ---------- 데이터 수집 함수 (기존과 동일) ----------
def fetch_ohlcv_10000(symbol: str, tf: str, end_ms=None) -> pd.DataFrame:
    interval = str(tf)
    if end_ms is None: end_ms = int(datetime.now(tz=timezone.utc).timestamp()*1000)
    rows = []
    while len(rows) < MAX_CANDLES:
        r = session.get_kline(category="linear", symbol=symbol, interval=interval, end=end_ms, limit=1000)
        if r.get("retCode") != 0 or not r["result"]["list"]: break
        lst = r["result"]["list"]
        for it in lst:
            rows.append((int(it[0]), float(it[1]), float(it[2]), float(it[3]), float(it[4]), float(it[5])))
        end_ms = min(int(x[0]) for x in lst) - 1
        if len(lst) < 1000: break
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True); df.reset_index(drop=True, inplace=True)
    return df

# ---------- 시뮬레이션 엔진 ----------
def run_simulation(symbol, tf, rsi_p, ema_p, tp_roe, sl_roe):
    ohlc = fetch_ohlcv_10000(symbol, tf, int(datetime.now(tz=timezone.utc).timestamp()*1000))
    if ohlc.empty: return
    
    # 지표 계산
    delta = ohlc['close'].diff()
    up = delta.clip(lower=0); down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/rsi_p, adjust=False).mean()
    roll_down = down.ewm(alpha=1/rsi_p, adjust=False).mean()
    ohlc['rsi'] = 100 - (100/(1 + roll_up/(roll_down + 1e-12)))
    ohlc['ema_filter'] = ohlc['close'].ewm(span=ema_p, adjust=False).mean()
    
    log = []
    position = None 
    entry_px, qty, init_margin = 0, 0, 0

    for i in range(1, len(ohlc)):
        row = ohlc.iloc[i]
        prev = ohlc.iloc[i-1]
        px, rv, ema = row['close'], row['rsi'], row['ema_filter']
        
        if np.isnan(rv) or np.isnan(ema): continue

        # 1. 진입 로직
        if position is None:
            # 상승 추세 + RSI 30 상향 돌파
            if px > ema and prev['rsi'] < OPEN_LONG_RSI and rv >= OPEN_LONG_RSI:
                position = "LONG"; entry_px = px
                qty = (EQUITY * LEVERAGE) / entry_px
                init_margin = (EQUITY * LEVERAGE) / LEVERAGE
                
            # 하락 추세 + RSI 70 하향 돌파
            elif px < ema and prev['rsi'] > OPEN_SHORT_RSI and rv <= OPEN_SHORT_RSI:
                position = "SHORT"; entry_px = px
                qty = (EQUITY * LEVERAGE) / entry_px
                init_margin = (EQUITY * LEVERAGE) / LEVERAGE

        # 2. 청산 로직
        elif position is not None:
            unreal = (px - entry_px) * qty if position == "LONG" else (entry_px - px) * qty
            roe = (unreal / init_margin) * 100

            msg = ""
            if roe >= tp_roe: msg = "TP"
            elif roe <= -sl_roe: msg = "SL"

            if msg:
                dt = datetime.fromtimestamp(row['ts']//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                log.append([dt, symbol, tf, px, rv, "CLOSE", msg, entry_px, unreal, roe])
                position = None

    if log:
        res_df = pd.DataFrame(log, columns=["datetime","symbol","timeframe","close","rsi","pos","msg","entry","pnl","roe"])
        os.makedirs(OUT_DIR, exist_ok=True)
        fname = f"FINAL_{symbol}_{tf}_EMA{ema_p}_R{rsi_p}_T{tp_roe}_S{sl_roe}.csv"
        path = os.path.join(OUT_DIR, fname)
        res_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장: {fname} (매매: {len(log)}회)")

if __name__ == "__main__":
    print(f"--- 다중 파라미터 백테스팅 시작 ---")
    for s in SYMBOL:
        for tf in TIMEFRAME:
            for ep in EMA_PERIOD_ARR:  # EMA 배열 루프 추가
                for rp in RSI_PERIOD:
                    for tp in TP_ROE_ARR:
                        for sl in SL_ROE_ARR:
                            run_simulation(s, tf, rp, ep, tp, sl)