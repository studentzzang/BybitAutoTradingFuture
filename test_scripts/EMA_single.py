import os, time, math, re, glob
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ================= 사용자 설정 =================
OUT_DIR        = r"D:\Projects\AutoCoinAI\test"
SYMBOLS        = ["PUMPFUNUSDT"]
TIMEFRAMES     = ["3","5","30","60"]  # "1","3","5","15","30","60",...,"D","W","M"

# === EMA 1개 시스템 파라미터 그리드 ===
EMA_PERIOD_ARR       = [100, 200]

# 횡보 필터(선택): EMA 기울기 + EMA/가격 거리
USE_SLOPE_FILTER     = True
SLOPE_LOOKBACK_ARR   = [8, 10, 12]          # n봉 전 EMA 대비
SLOPE_THRESH_ARR     = [0.0008, 0.0010]     # |(EMA-EMA[n])/EMA| > thresh

USE_DIST_FILTER      = True
DIST_THRESH_ARR      = [0.0010, 0.0015]     # |(C-EMA)/EMA| > thresh

# TP/SL (ROE%)
TP_ROE_ARR     = [5, 7.5, 10]
SL_ROE_ARR     = [5, 7.5, 10]

# 백테스트 환경
EQUITY         = 100.0
LEVERAGE       = 5
START          = "2025-01-01"   # UTC
END            = None
MAX_CANDLES    = 20000
SLEEP_PER_REQ  = 0.12
MAX_RETRY      = 3

# 수수료/슬리피지 (bps)
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
            if ts < start_ms: 
                continue
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

# ================= 백테스트 (EMA 1개 시스템) =================
def backtest_ema1(symbol: str, tf: str,
                  ema_period: int,
                  slope_lb: Optional[int],
                  slope_th: Optional[float],
                  dist_th: Optional[float],
                  tp_roe: float, sl_roe: float,
                  start_ms: Optional[int], end_ms: Optional[int]) -> pd.DataFrame:

    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        return pd.DataFrame(columns=[
            "datetime","symbol","timeframe","ema_p","slope_lb","slope_th","dist_th",
            "포지션","비고","entry_price","exit_price","미실현PnL","ROE"
        ])

    close = ohlc["close"].astype(float)
    ohlc["ema"] = ema(close, ema_period)

    # 필터 계산 (NaN 안전)
    if USE_SLOPE_FILTER and slope_lb is not None and slope_th is not None:
        slope_pct = (ohlc["ema"] - ohlc["ema"].shift(slope_lb)) / ohlc["ema"]
        slope_ok = (slope_pct.abs() > slope_th).fillna(False)
    else:
        slope_ok = pd.Series([True]*len(ohlc))

    if USE_DIST_FILTER and dist_th is not None:
        dist = (close - ohlc["ema"]).abs() / ohlc["ema"]
        dist_ok = (dist > dist_th).fillna(False)
    else:
        dist_ok = pd.Series([True]*len(ohlc))

    # 신호: EMA 위면 롱, 아래면 숏 (동률은 거래 안함)
    above = (close > ohlc["ema"]).fillna(False)
    below = (close < ohlc["ema"]).fillna(False)

    # 거래 가능 여부(횡보 필터 통과)
    tradable = (slope_ok & dist_ok).fillna(False)

    position: Optional[str] = None
    entry_px: Optional[float] = None
    qty: Optional[float] = None

    notional = EQUITY * LEVERAGE
    fee = TAKER_FEE_BPS / 10_000.0
    slip = SLIPPAGE_BPS  / 10_000.0

    cols = ["datetime","symbol","timeframe","ema_p","slope_lb","slope_th","dist_th",
            "포지션","비고","entry_price","exit_price","미실현PnL","ROE"]
    log_rows: List[List] = []

    # 워밍업: EMA + slope_lb 고려
    warm = ema_period + 2
    if USE_SLOPE_FILTER and slope_lb is not None:
        warm = max(warm, slope_lb + ema_period + 2)
    start_idx = min(max(warm, 5), len(ohlc)-1)

    for i in range(start_idx, len(ohlc)):
        ts = int(ohlc.loc[i, "ts"])
        dt = datetime.fromtimestamp(ts//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px_close = float(ohlc.loc[i, "close"])

        can_trade = bool(tradable.iloc[i])

        # === 진입 (종가 기준, 기존 방식 유지) ===
        if position is None and can_trade:
            if bool(above.iloc[i]):
                position = "LONG"
                entry_px = px_close * (1 + slip)
                qty = notional / entry_px
                continue
            elif bool(below.iloc[i]):
                position = "SHORT"
                entry_px = px_close * (1 - slip)
                qty = notional / entry_px
                continue

        # === 청산 (종가 기준) ===
        if position == "LONG":
            exit_px = px_close * (1 - slip)
            pnl = (exit_px - entry_px) * qty - 2*fee*notional
            roe_pct = (pnl / EQUITY) * 100.0

            if roe_pct >= tp_roe:
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "TP LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if roe_pct <= -sl_roe:
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "SL LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            # 반대 신호(EMA 아래로)면 청산
            if bool(below.iloc[i]):
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "XC LONG", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue

        elif position == "SHORT":
            exit_px = px_close * (1 + slip)
            pnl = (entry_px - exit_px) * qty - 2*fee*notional
            roe_pct = (pnl / EQUITY) * 100.0

            if roe_pct >= tp_roe:
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "TP SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            if roe_pct <= -sl_roe:
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "SL SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue
            # 반대 신호(EMA 위로)면 청산
            if bool(above.iloc[i]):
                log_rows.append([dt, symbol, tf, ema_period, slope_lb, slope_th, dist_th,
                                 "CLOSE", "XC SHORT", entry_px, exit_px, pnl, roe_pct])
                position = None; entry_px = None; qty = None
                continue

    return pd.DataFrame(log_rows, columns=cols)

# ================= Summary 생성 =================
def build_summary(out_dir: str):
    pattern = os.path.join(out_dir, "*_EMA1_*.csv")
    files = sorted(glob.glob(pattern))

    rows = []
    # 파일명 예시:
    # PUMPFUNUSDT_30_EMA1_EMA200_SLP10-0.001_DST0.0015_TP7.5_SL10.csv
    fname_re = re.compile(
        r"^(?P<symbol>[A-Z0-9]+)_(?P<tf>[0-9DWM]+)_EMA1_"
        r"EMA(?P<ema>\d+)_"
        r"SLP(?P<slb>\d+)-(?P<sth>[\d\.]+)_"
        r"DST(?P<dst>[\d\.]+)_"
        r"TP(?P<tp>[\d\.]+)_SL(?P<sl>[\d\.]+)\.csv$"
    )

    for fpath in files:
        fname = os.path.basename(fpath)
        m = fname_re.match(fname)
        meta = {"symbol":None,"timeframe":None,"ema_p":None,"slope_lb":None,"slope_th":None,"dist_th":None,"tp":None,"sl":None,"file":fname}
        if m:
            meta.update({
                "symbol": m.group("symbol"),
                "timeframe": m.group("tf"),
                "ema_p": int(m.group("ema")),
                "slope_lb": int(m.group("slb")),
                "slope_th": float(m.group("sth")),
                "dist_th": float(m.group("dst")),
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
        "symbol","timeframe","ema_p","slope_lb","slope_th","dist_th","tp","sl","file",
        "trades","total_pnl_usdt","total_roe_pct","win_rate_pct","min_loss_pnl_usdt","min_roe_pct",
        "first_trade_at","last_trade_at"
    ])
    if not summary_df.empty:
        summary_df.sort_values(["symbol","timeframe","ema_p","slope_lb","slope_th","dist_th","tp","sl","last_trade_at"], inplace=True)

    summary_path = os.path.join(out_dir, "summary_EMA1.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] summary_EMA1.csv saved -> {summary_path}")

# ================= 실행 =================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    start_ms = parse_date_ms(START)
    end_ms   = parse_date_ms(END)

    # 필터 파라미터 그리드 구성
    slope_grid = [(None, None)] if not USE_SLOPE_FILTER else [(lb, th) for lb in SLOPE_LOOKBACK_ARR for th in SLOPE_THRESH_ARR]
    dist_grid  = [None] if not USE_DIST_FILTER else list(DIST_THRESH_ARR)

    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            for ema_p in EMA_PERIOD_ARR:
                for (slb, sth) in slope_grid:
                    for dst in dist_grid:
                        for tp in TP_ROE_ARR:
                            for sl in SL_ROE_ARR:
                                # 파일명 안정화를 위해 None이면 기본 표기값 부여
                                slb_v = slb if slb is not None else 0
                                sth_v = sth if sth is not None else 0.0
                                dst_v = dst if dst is not None else 0.0

                                try:
                                    trades_df = backtest_ema1(s, tf, ema_p, slb, sth, dst, tp, sl, start_ms, end_ms)
                                except Exception as e:
                                    print(f"[SKIP] {s}_{tf}_EMA1_EMA{ema_p}_SLP{slb_v}-{sth_v}_DST{dst_v}_TP{tp}_SL{sl}: {e}")
                                    continue

                                fname = f"{s}_{tf}_EMA1_EMA{ema_p}_SLP{slb_v}-{sth_v}_DST{dst_v}_TP{tp}_SL{sl}.csv"
                                fpath = os.path.join(OUT_DIR, fname)
                                trades_df.to_csv(fpath, index=False, encoding="utf-8-sig")
                                print(f"✅ 저장: {fpath}")

    build_summary(OUT_DIR)
    print("✅ 완료")
