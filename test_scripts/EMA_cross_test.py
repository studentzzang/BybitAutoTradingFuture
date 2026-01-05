
import os, time, math, re, glob
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ================= 사용자 설정 =================
OUT_DIR        = r"D:\Projects\AutoCoinAI\test"   # 결과 저장 폴더
SYMBOLS        = ["PUMPFUNUSDT"]
TIMEFRAMES     = [1,5]                       # Bybit interval: "1","3","5","15","30","60",...,"D","W","M"

EMA_FAST_ARR   = [5, 7, 12,20,50]
EMA_SLOW_ARR   = [9, 14,26,50]

TP_ROE_ARR     = [3]                    # ROE% 목표
SL_ROE_ARR     = [5,10]                    # ROE% 손절

EQUITY         = 100.0                            # 증거금(USDT)
LEVERAGE       = 10
START          = "2025-03-01"                     # 시작일 (UTC)
END            = None                              # None이면 현재
MAX_CANDLES    = 20000
SLEEP_PER_REQ  = 0.12
MAX_RETRY      = 3

# 수수료/슬리피지 (요청: 0)
TAKER_FEE_BPS  = 0.0
SLIPPAGE_BPS   = 0.0

# ================= Bybit HTTP =================
session = HTTP() 

def parse_date_ms(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def bybit_interval(tf: str) -> str:
    tf = str(tf).upper()
    mapping = {
        "1":"1","3":"3","5":"5","15":"15","30":"30","60":"60","120":"120","240":"240","360":"360","720":"720",
        "D":"D","W":"W","M":"M"
    }
    if tf not in mapping:
        raise ValueError(f"unsupported timeframe: {tf}")
    return mapping[tf]

def fetch_ohlcv(symbol: str, tf: str, start_ms: Optional[int], end_ms: Optional[int], cap: Optional[int]) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if start_ms is None: start_ms = parse_date_ms("2018-01-01")
    if end_ms   is None:
        end_ms   = int(datetime.now(tz=timezone.utc).timestamp()*1000)

    rows: List[Tuple[int,float,float,float,float,float]] = []
    hard_cap = cap if cap is not None else 10**12
    cur_end = end_ms
    last_exc = None

    while len(rows) < hard_cap and cur_end > start_ms:
        req_limit = int(min(1000, hard_cap - len(rows)))
        for _ in range(MAX_RETRY):
            try:
                resp = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    end=cur_end,
                    limit=req_limit
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                time.sleep(0.3)
        if last_exc is not None:
            raise RuntimeError(f"Bybit API error: {last_exc}")

        if resp.get("retCode") != 0:
            raise RuntimeError(resp.get("retMsg","bybit error"))

        lst = resp.get("result",{}).get("list",[])
        if not lst:
            break

        for it in lst:
            ts = int(it[0])
            if ts < start_ms: continue
            o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4]); v = float(it[5])
            rows.append((ts,o,h,l,c,v))

        min_ts = min(int(x[0]) for x in lst)
        cur_end = min_ts - 1

        if len(lst) < req_limit:
            break
        time.sleep(SLEEP_PER_REQ)

    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    if cap is not None:
        df = df.tail(int(cap))
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ================= 지표 =================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ================= 백테스트 =================
def backtest(symbol: str, tf: str, fast: int, slow: int, tp_roe: float, sl_roe: float,
             start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:
    assert fast < slow
    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        return pd.DataFrame(columns=[
            "datetime","symbol","timeframe","fast","slow","rsi_p","doorstep",
            "포지션","비고","entry_price","exit_price","미실현PnL","ROE"
        ])

    close = ohlc["close"].astype(float)
    ohlc["ema_fast"] = ema(close, fast)
    ohlc["ema_slow"] = ema(close, slow)

    # 교차 플래그 (NaN 안전)
    above = (ohlc["ema_fast"] > ohlc["ema_slow"]).fillna(False)
    above_prev = above.shift(1).fillna(False)
    cross_up = (~above_prev) & (above)     # 골든 → LONG 진입
    cross_dn = (above_prev) & (~above)     # 데드  → SHORT 진입

    position: Optional[str] = None
    entry_px: Optional[float] = None
    qty: Optional[float] = None

    notional = EQUITY * LEVERAGE
    fee = TAKER_FEE_BPS / 10_000.0
    slip = SLIPPAGE_BPS  / 10_000.0

    cols = ["datetime","symbol","timeframe","fast","slow","rsi_p","doorstep",
            "포지션","비고","entry_price","exit_price","미실현PnL","ROE"]
    log_rows: List[List] = []

    start_idx = max(fast, slow) + 1
    n = len(ohlc)

    for i in range(start_idx, n):
        ts = int(ohlc.loc[i,"ts"])
        dt = datetime.fromtimestamp(ts//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i,"close"])

        # 진입
        if position is None:
            if bool(cross_up.iloc[i]):
                position = "LONG"
                entry_px = px * (1 + slip)
                qty = notional / entry_px
                continue
            elif bool(cross_dn.iloc[i]):
                position = "SHORT"
                entry_px = px * (1 - slip)
                qty = notional / entry_px
                continue

        # 청산
        if position == "LONG":
            exit_px = px * (1 - slip)
            pnl = (exit_px - entry_px) * qty - 2*fee*notional
            roe_pct = (pnl / EQUITY) * 100.0

            if roe_pct >= tp_roe:
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "TP LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if roe_pct <= -sl_roe:
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "SL LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if bool(cross_dn.iloc[i]):
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "XC LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue

        elif position == "SHORT":
            exit_px = px * (1 + slip)
            pnl = (entry_px - exit_px) * qty - 2*fee*notional
            roe_pct = (pnl / EQUITY) * 100.0

            if roe_pct >= tp_roe:
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "TP SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if roe_pct <= -sl_roe:
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "SL SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if bool(cross_up.iloc[i]):
                log_rows.append([dt, symbol, tf, fast, slow, 0, 0, "CLOSE", "XC SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue

    return pd.DataFrame(log_rows, columns=cols)

# ================= Summary 생성 =================
def build_summary(out_dir: str):
    pattern = os.path.join(out_dir, "*_EMA*-*_TP*_*SL*.csv")
    files = sorted(glob.glob(pattern))

    rows = []
    fname_re = re.compile(
        r"^(?P<symbol>[A-Z0-9]+)_(?P<tf>[0-9DWM]+)_EMA(?P<fast>\d+)-(?P<slow>\d+)_TP(?P<tp>[\d\.]+)_SL(?P<sl>[\d\.]+)\.csv$"
    )

    for fpath in files:
        fname = os.path.basename(fpath)
        m = fname_re.match(fname)
        meta = {"symbol":None,"timeframe":None,"fast":None,"slow":None,"tp":None,"sl":None,"file":fname}
        if m:
            meta.update({
                "symbol": m.group("symbol"),
                "timeframe": m.group("tf"),
                "fast": int(m.group("fast")),
                "slow": int(m.group("slow")),
                "tp": float(m.group("tp")),
                "sl": float(m.group("sl")),
            })

        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(fpath, encoding="utf-8")
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            rows.append({**meta,
                "trades": 0,
                "total_pnl_usdt": 0.0,
                "total_roe_pct": 0.0,
                "win_rate_pct": 0.0,
                "min_loss_pnl_usdt": 0.0,
                "min_roe_pct": 0.0,
                "first_trade_at": None,
                "last_trade_at": None
            })
            continue

        trades = len(df)
        total_pnl_usdt = float(df["미실현PnL"].sum()) if "미실현PnL" in df.columns else 0.0
        total_roe_pct  = float(df["ROE"].sum()) if "ROE" in df.columns else 0.0
        win_rate_pct   = float((df["ROE"] > 0).mean() * 100.0) if "ROE" in df.columns and trades>0 else 0.0
        min_loss_pnl   = float(df["미실현PnL"].min()) if "미실현PnL" in df.columns else 0.0
        min_roe_pct    = float(df["ROE"].min()) if "ROE" in df.columns else 0.0
        first_trade_at = df["datetime"].min() if "datetime" in df.columns else None
        last_trade_at  = df["datetime"].max() if "datetime" in df.columns else None

        rows.append({**meta,
            "trades": trades,
            "total_pnl_usdt": round(total_pnl_usdt, 6),
            "total_roe_pct":  round(total_roe_pct, 4),
            "win_rate_pct":   round(win_rate_pct, 2),
            "min_loss_pnl_usdt": round(min_loss_pnl, 6),
            "min_roe_pct":    round(min_roe_pct, 4),
            "first_trade_at": first_trade_at,
            "last_trade_at":  last_trade_at
        })

    summary_df = pd.DataFrame(rows, columns=[
        "symbol","timeframe","fast","slow","tp","sl","file",
        "trades","total_pnl_usdt","total_roe_pct","win_rate_pct","min_loss_pnl_usdt","min_roe_pct",
        "first_trade_at","last_trade_at"
    ])
    if not summary_df.empty:
        summary_df.sort_values(["symbol","timeframe","fast","slow","tp","sl","last_trade_at"], inplace=True)
    summary_path = os.path.join(out_dir, "summary_EMA.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] summary_EMA.csv saved -> {summary_path}")

# ================= 실행 =================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    start_ms = parse_date_ms(START)
    end_ms   = parse_date_ms(END)

    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            for fast in EMA_FAST_ARR:
                for slow in EMA_SLOW_ARR:
                    if fast >= slow: 
                        continue
                    for tp in TP_ROE_ARR:
                        for sl in SL_ROE_ARR:
                            try:
                                trades_df = backtest(s, tf, fast, slow, tp, sl, start_ms, end_ms)
                            except Exception as e:
                                print(f"[SKIP] {s}_{tf}_EMA{fast}-{slow}_TP{tp}_SL{sl}: {e}")
                                continue

                            fname = f"{s}_{tf}_EMA{fast}-{slow}_TP{tp}_SL{sl}.csv"
                            fpath = os.path.join(OUT_DIR, fname)
                            trades_df.to_csv(fpath, index=False, encoding="utf-8-sig")
                            print(f"✅ 저장: {fpath}")

    # summary 생성
    build_summary(OUT_DIR)
    print("✅ 완료")
