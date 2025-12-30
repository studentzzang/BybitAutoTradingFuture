import os
import sys
import time
import signal # 중지 신호 처리를 위해 추가
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# ====== 사용자 설정 변수 ======
SYMBOL = ["PUMPFUNUSDT"] 
LEVERAGE = 5
TIMEFRAME = ["5", "15", "30"] 
RSI_PERIOD = [14]
EQUITY = 100.0
START = "2025-01-01"
END = "2025-12-28"
OUT_DIR = "test" # 현재 폴더 아래 test 폴더에 저장

ATR_PERIOD_ARR = [9, 14, 20]
TP_ATR_MULT_ARR = [3.0, 4.5]
SL_ATR_MULT_ARR = [1.5, 2.0]

OPEN_SHORT_RSI = 70.0
OPEN_LONG_RSI  = 30.0
MAX_CANDLES = 20000 

session = HTTP()

# ---------- Ctrl+C 중지 처리 ----------
def signal_handler(sig, frame):
    print("\n[!] 사용자에 의해 중지 신호(Ctrl+C)가 감지되었습니다. 프로그램을 종료합니다.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ---------- 유틸리티 ----------
def parse_date(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try:
        dt = datetime.fromisoformat(s)
    except:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def bybit_interval(tf: str) -> str:
    mapping = {"1":"1","3":"3","5":"5","15":"15","30":"30","60":"60","D":"D"}
    return mapping.get(tf, "15")

def fetch_ohlcv_10000(symbol: str, tf: str, start_ms=None, end_ms=None) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp()*1000)

    rows = []
    curr_end = end_ms
    print(f" > {symbol} 데이터 불러오는 중...", end="", flush=True)
    
    while len(rows) < MAX_CANDLES:
        try:
            r = session.get_kline(category="linear", symbol=symbol, interval=interval, end=curr_end, limit=1000)
            if r.get("retCode") != 0: break
            lst = r["result"]["list"]
            if not lst: break
            for it in lst:
                ts = int(it[0]); o,h,l,c,v = map(float, it[1:6])
                rows.append((ts,o,h,l,c,v))
            curr_end = min(int(x[0]) for x in lst) - 1
            if len(lst) < 1000: break
            time.sleep(0.05)
        except: break

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f" 완료 ({len(df)}개)")
    return df

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    roll_up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    roll_down = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ---------- 시뮬레이션 ----------
def run_simulation(symbol, tf, rsi_p, atr_p, tp_m, sl_m):
    ohlc = fetch_ohlcv_10000(symbol, tf, parse_date(START), parse_date(END))
    if ohlc.empty:
        print(f" [!] {symbol} 데이터를 가져오지 못했습니다. 심볼명을 확인하세요.")
        return 
    
    ohlc["rsi"] = compute_rsi(ohlc["close"], rsi_p)
    ohlc["atr"] = compute_atr(ohlc, atr_p)
    
    log = []
    position = None 
    entry_px, stop_loss, take_profit, qty = 0, 0, 0, 0

    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        px, rv, av = row['close'], row['rsi'], row['atr']
        if np.isnan(rv) or np.isnan(av): continue

        if position is None:
            if rv <= OPEN_LONG_RSI:
                position = "LONG"; entry_px = px
                stop_loss = entry_px - (av * sl_m)
                take_profit = entry_px + (av * tp_m)
                qty = (EQUITY * LEVERAGE) / entry_px
            elif rv >= OPEN_SHORT_RSI:
                position = "SHORT"; entry_px = px
                stop_loss = entry_px + (av * sl_m)
                take_profit = entry_px - (av * tp_m)
                qty = (EQUITY * LEVERAGE) / entry_px

        elif position == "LONG":
            if px >= take_profit or px <= stop_loss:
                res = "TP" if px >= take_profit else "SL"
                pnl = (px - entry_px) * qty
                dt = datetime.fromtimestamp(row['ts']//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                log.append([dt, symbol, tf, px, rv, "CLOSE", res, entry_px, pnl, (pnl/EQUITY)*100])
                position = None
        elif position == "SHORT":
            if px <= take_profit or px >= stop_loss:
                res = "TP" if px <= take_profit else "SL"
                pnl = (entry_px - px) * qty
                dt = datetime.fromtimestamp(row['ts']//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                log.append([dt, symbol, tf, px, rv, "CLOSE", res, entry_px, pnl, (pnl/EQUITY)*100])
                position = None

    if log:
        res_df = pd.DataFrame(log, columns=["datetime","symbol","timeframe","close","rsi","pos","msg","entry","pnl","roe"])
        if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
        fname = f"{symbol}_{tf}_ATR{atr_p}_T{tp_m}_S{sl_m}.csv"
        path = os.path.join(OUT_DIR, fname)
        res_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  >> 저장 성공: {path}")
    else:
        print("  >> 조건에 맞는 매매 기록이 없습니다.")

if __name__ == "__main__":
    print(f"--- 백테스팅 시작 (저장경로: {os.path.abspath(OUT_DIR)}) ---")
    for s in SYMBOL:
        for tf in TIMEFRAME:
            for ap in ATR_PERIOD_ARR:
                for tp in TP_ATR_MULT_ARR:
                    for sl in SL_ATR_MULT_ARR:
                        print(f"테스트: {s}/{tf}분봉 (ATR:{ap}, TP:{tp}, SL:{sl})")
                        run_simulation(s, tf, 14, ap, tp, sl)