
import os, time, math
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ================= 사용자 설정 =================
OUT_DIR        = r"d:\Projects\AutoCoinAI\test"
SYMBOLS        = ["DOGEUSDT", "PUMPFUNUSDT", "ETHUSDT"]
TIMEFRAMES     = ["5"]                    # Bybit 인터벌 문자열(아래 bybit_interval 참고)

EMA_FAST_ARR   = [5, 8]
EMA_SLOW_ARR   = [13]

TP_ROE_ARR     = [2.5, 5, 7.5,10]                 # ROE% (레버리지 반영된 수익률)
SL_ROE_ARR     = [2.5, 5, 7.5,10]                 # ROE%

EQUITY         = 100.0                         # 계정 기준 증거금(USDT) 가정
LEVERAGE       = 5
START          = "2025-03-01"                  # ISO 또는 "YYYY-MM-DD"
END            = None                          # None이면 현재시각
MAX_CANDLES    = 20000
SLEEP_PER_REQ  = 0.15                          # Bybit rate-limit 완화
MAX_RETRY      = 3                             # API 실패 재시도 횟수

# ================= Bybit HTTP =================
session = HTTP()  # (주의) 환경에 따라 key/secret 지정 필요할 수 있음

def parse_date(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def bybit_interval(tf: str) -> str:
    tf = str(tf).upper()
    # Bybit UTA 인터벌: "1","3","5","15","30","60","120","240","360","720","D","W","M"
    mapping = {
        "1": "1", "3": "3", "5": "5", "15": "15", "30": "30", "60": "60",
        "120": "120", "240": "240", "360": "360", "720": "720",
        "D": "D", "W": "W", "M": "M"
    }
    if tf not in mapping:
        raise ValueError(f"unsupported tf: {tf}")
    return mapping[tf]

def fetch_ohlcv(symbol: str, tf: str, start_ms: Optional[int], end_ms: Optional[int], cap: Optional[int]) -> pd.DataFrame:
    """Bybit unified trading API에서 선물(Linear) kline 끊어서 가져오기 (끝에서 과거로 역방향 페이징)"""
    interval = bybit_interval(tf)
    if start_ms is None:
        start_ms = parse_date("2018-01-01")
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows: List[Tuple[int, float, float, float, float, float]] = []
    hard_cap = cap if cap is not None else 10**12
    cur_end = end_ms

    while len(rows) < hard_cap and cur_end > start_ms:
        req_limit = int(min(1000, hard_cap - len(rows)))

        # 재시도 루프
        last_exc = None
        for _ in range(MAX_RETRY):
            try:
                resp = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    end=cur_end,
                    limit=req_limit,
                )
                break
            except Exception as e:
                last_exc = e
                time.sleep(0.3)
        if last_exc is not None and 'resp' not in locals():
            raise RuntimeError(f"Bybit API error: {last_exc}")

        if resp.get("retCode") != 0:
            raise RuntimeError(resp.get("retMsg", "bybit error"))

        lst = resp.get("result", {}).get("list", [])
        if not lst:
            break

        for it in lst:
            ts = int(it[0])
            if ts < start_ms:
                continue
            o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4]); v = float(it[5])
            rows.append((ts, o, h, l, c, v))

        # 다음 페이지: 가장 과거 ts보다 1ms 앞
        min_ts = min(int(x[0]) for x in lst)
        cur_end = min_ts - 1

        if len(lst) < req_limit:
            break
        time.sleep(SLEEP_PER_REQ)

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    if cap is not None:
        df = df.tail(int(cap))
    df.reset_index(drop=True, inplace=True)
    return df

# ================= 지표 =================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# (호환용: 안 써도 열은 남김)
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

# ================= 백테스트 =================
# rsi_p, doorstep은 파일 호환만 위해 0으로 기록
def backtest(symbol: str, tf: str, fast: int, slow: int, rsi_p: int, doorstep: float,
             tp_roe: float, sl_roe: float) -> pd.DataFrame:
    assert fast < slow, "fast < slow 이어야 합니다."
    start_ms = parse_date(START); end_ms = parse_date(END)

    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        raise SystemExit(f"[{symbol}@{tf}] no data")

    # 지표
    ohlc["ema_fast"] = ema(ohlc["close"], fast)
    ohlc["ema_slow"] = ema(ohlc["close"], slow)

    # 교차 판정
    fast_gt = ohlc["ema_fast"] > ohlc["ema_slow"]
    fast_gt_prev = fast_gt.shift(1).fillna(False)
    cross_up = (~fast_gt_prev) & (fast_gt)      # 골든 → 롱 전환
    cross_dn = (fast_gt_prev) & (~fast_gt)      # 데드 → 숏 전환

    position: Optional[str] = None
    entry_px: Optional[float] = None
    qty: Optional[float] = None

    notional = EQUITY * LEVERAGE               # 포지션 명목 가치(USDT)

    cols = ["datetime","symbol","timeframe","fast","slow","rsi_p","doorstep",
            "포지션","비고","entry_price","exit_price","미실현PnL","ROE"]
    log_rows: List[List] = []

    for i in range(len(ohlc)):
        ts = int(ohlc.loc[i, "ts"]) // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i, "close"])

        # 진입
        if position is None:
            if cross_up.iloc[i]:
                position = "LONG"
                entry_px = px
                qty = notional / entry_px
                # (진입 자체는 기록 안 함 — 청산 시점만 기록)
                continue
            elif cross_dn.iloc[i]:
                position = "SHORT"
                entry_px = px
                qty = notional / entry_px
                continue

        # 청산 (ROE% TP/SL or 반대 교차)
        if position is not None:
            if position == "LONG":
                pnl = (px - entry_px) * qty
                equity_used = notional / LEVERAGE  # = EQUITY
                roe_pct = (pnl / equity_used) * 100.0

                if roe_pct >= tp_roe:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "TP LONG", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue
                if roe_pct <= -sl_roe:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "SL LONG", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue
                if cross_dn.iloc[i]:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "XC LONG", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue

            elif position == "SHORT":
                pnl = (entry_px - px) * qty
                equity_used = notional / LEVERAGE
                roe_pct = (pnl / equity_used) * 100.0

                if roe_pct >= tp_roe:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "TP SHORT", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue
                if roe_pct <= -sl_roe:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "SL SHORT", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue
                if cross_up.iloc[i]:
                    log_rows.append([dt, symbol, tf, fast, slow, 0, 0,
                                     "CLOSE", "XC SHORT", entry_px, px, pnl, roe_pct])
                    position = None; entry_px = None; qty = None
                    continue

    trades_df = pd.DataFrame(log_rows, columns=cols)
    return trades_df

# ================= 실행 =================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            for fast in EMA_FAST_ARR:
                for slow in EMA_SLOW_ARR:
                    if fast >= slow:
                        continue
                    for tp in TP_ROE_ARR:
                        for sl in SL_ROE_ARR:
                            try:
                                df = backtest(s, tf, fast, slow, 0, 0, tp, sl)
                            except SystemExit as e:
                                print(f"[SKIP] {s}_{tf}_EMA{fast}-{slow}_TP{tp}_SL{sl}: {e}")
                                continue
                            except Exception as e:
                                print(f"[ERR ] {s}_{tf}_EMA{fast}-{slow}_TP{tp}_SL{sl}: {e}")
                                continue

                            fname = f"{s}_{tf}_EMA{fast}-{slow}_TP{tp}_SL{sl}.csv"
                            fpath = os.path.join(OUT_DIR, fname)
                            df.to_csv(fpath, index=False, encoding="utf-8-sig")
                            print(f"✅ 저장: {fpath}")
