# 이 코드는 청산, 진입 부분만 저장함

import os, time
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ================= 사용자 설정 =================
OUT_DIR = r"d:\Projects\AutoCoinAI\tests" #왜 상대경로가 안 돼 ㅠ
SYMBOLS        = ["BTCUSDT","DOGEUSDT", "PUMPFUNUSDT", "ETHUSDT"]
TIMEFRAMES     = []                 # 1H 기본
EMA_FAST_ARR   = [5, 8]
EMA_SLOW_ARR   = [13, 15]
EMA_PASS_GAP   = 35 #SLOW FAST이만큼차이나면 패스
RSI_PERIODS    = [12, 18, 24] 
DOORSTEP_ARR   = [5, 10];
EQUITY         = 100.0
LEVERAGE       = 5
START          = "2025-01-01"
END            = None
MAX_CANDLES    = 20000
SLEEP_PER_REQ  = 0.12

# ================= Bybit HTTP =================
session = HTTP()

def parse_date(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()*1000)

def bybit_interval(tf: str) -> str:
    tf = str(tf).upper()
    mapping = {"1":"1","3":"3","5":"5","15":"15","30":"30","60":"60","120":"120","240":"240","360":"360","720":"720","D":"D","W":"W","M":"M"}
    if tf not in mapping: raise ValueError(f"unsupported tf: {tf}")
    return mapping[tf]

def fetch_ohlcv(symbol: str, tf: str, start_ms: Optional[int], end_ms: Optional[int], cap: Optional[int]) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if start_ms is None: start_ms = parse_date("2018-01-01")
    if end_ms is None:   end_ms   = int(datetime.now(tz=timezone.utc).timestamp()*1000)

    rows: List[Tuple[int,float,float,float,float,float]] = []
    hard_cap = cap if cap is not None else 10**12
    cur_end = end_ms

    while len(rows) < hard_cap and cur_end > start_ms:
        req_limit = int(min(1000, hard_cap - len(rows)))
        resp = session.get_kline(category="linear", symbol=symbol, interval=interval, end=cur_end, limit=req_limit)
        if resp.get("retCode") != 0:
            raise RuntimeError(resp.get("retMsg", "bybit error"))
        lst = resp.get("result", {}).get("list", [])
        if not lst: break

        for it in lst:
            ts = int(it[0])
            if ts < start_ms: continue
            o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4]); v = float(it[5])
            rows.append((ts,o,h,l,c,v))

        min_ts = min(int(x[0]) for x in lst)
        cur_end = min_ts - 1
        if len(lst) < req_limit: break
        time.sleep(SLEEP_PER_REQ)

    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    if cap is not None:
        df = df.tail(int(cap))
    df.reset_index(drop=True, inplace=True)
    return df

# ================= 지표 =================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

# ================= 시뮬 =================
def backtest(symbol: str, tf: str, fast: int, slow: int, rsi_p: int, doorstep: float) -> pd.DataFrame:
    assert fast < slow
    start_ms = parse_date(START); end_ms = parse_date(END)
    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        raise SystemExit(f"[{symbol}@{tf}] no data")

    ohlc["ema_fast"] = ema(ohlc["close"], fast)
    ohlc["ema_slow"] = ema(ohlc["close"], slow)
    ohlc["rsi"]      = compute_rsi(ohlc["close"], rsi_p)

    fast_gt_prev = ohlc["ema_fast"] > ohlc["ema_slow"]
    fast_gt_prev_shift = fast_gt_prev.shift(1).fillna(False) # 여기 경고떠도문제없
    cross_up = (~fast_gt_prev_shift) & (fast_gt_prev)
    cross_dn = (fast_gt_prev_shift) & (~fast_gt_prev)

    position=None; entry_px=None; qty=None; peak_rsi=None; trough_rsi=None
    notional=EQUITY*LEVERAGE

    cols=["datetime","symbol","timeframe","fast","slow","rsi_p","doorstep",
          "포지션","비고","entry_price","exit_price","미실현PnL","ROE"]
    log_rows=[]

    for i in range(len(ohlc)):
        ts = int(ohlc.loc[i,"ts"])//1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i,"close"])
        rv = float(ohlc.loc[i,"rsi"]) if not np.isnan(ohlc.loc[i,"rsi"]) else None

        if position is None and rv is not None:
            if cross_up.iloc[i]:
                position="LONG"; entry_px=px; qty=notional/entry_px
                peak_rsi=rv; trough_rsi=None
                continue
            elif cross_dn.iloc[i]:
                position="SHORT"; entry_px=px; qty=notional/entry_px
                trough_rsi=rv; peak_rsi=None
                continue

        if position is not None and rv is not None:
            if position=="LONG":
                peak_rsi = rv if peak_rsi is None else max(peak_rsi, rv)
                if peak_rsi-rv >= doorstep:  # RSI 되돌림 청산
                    pnl=(px-entry_px)*qty; roe=pnl/(notional/LEVERAGE)
                    log_rows.append([dt,symbol,tf,fast,slow,rsi_p,doorstep,
                                     "CLOSE","close LONG",entry_px,px,pnl,roe])
                    position=None; entry_px=None; qty=None; peak_rsi=None
                    continue
                if cross_dn.iloc[i]:  # 반대 교차 청산
                    pnl=(px-entry_px)*qty; roe=pnl/(notional/LEVERAGE)
                    log_rows.append([dt,symbol,tf,fast,slow,rsi_p,doorstep,
                                     "CLOSE","stop LONG",entry_px,px,pnl,roe])
                    position="SHORT"; entry_px=px; qty=notional/entry_px
                    trough_rsi=rv; peak_rsi=None
                    continue

            elif position=="SHORT":
                trough_rsi = rv if trough_rsi is None else min(trough_rsi, rv)
                if rv-trough_rsi >= doorstep:  # RSI 되돌림 청산
                    pnl=(entry_px-px)*qty; roe=pnl/(notional/LEVERAGE)
                    log_rows.append([dt,symbol,tf,fast,slow,rsi_p,doorstep,
                                     "CLOSE","close SHORT",entry_px,px,pnl,roe])
                    position=None; entry_px=None; qty=None; trough_rsi=None
                    continue
                if cross_up.iloc[i]:  # 반대 교차 청산
                    pnl=(entry_px-px)*qty; roe=pnl/(notional/LEVERAGE)
                    log_rows.append([dt,symbol,tf,fast,slow,rsi_p,doorstep,
                                     "CLOSE","stop SHORT",entry_px,px,pnl,roe])
                    position="LONG"; entry_px=px; qty=notional/entry_px
                    peak_rsi=rv; trough_rsi=None
                    continue

    trades_df=pd.DataFrame(log_rows, columns=cols)
    return trades_df

# ================= 실행 =================
if __name__=="__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            for fast in EMA_FAST_ARR:
                for slow in EMA_SLOW_ARR:
                    if fast>=slow or slow-fast>=EMA_PASS_GAP: continue
                    for rsi_p in RSI_PERIODS:
                        for ds in DOORSTEP_ARR:
                            trades_df=backtest(s,tf,fast,slow,rsi_p,ds)
                            fname=f"{s}_{tf}_RSI{rsi_p}_EMA{fast}-{slow}_DS{ds}.csv"
                            trades_path=os.path.join(OUT_DIR,fname)
                            trades_df.to_csv(trades_path,index=False,encoding="utf-8-sig")
                            print(f"✅ 저장: {trades_path}")
