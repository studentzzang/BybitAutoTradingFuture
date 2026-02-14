#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ================= 사용자 설정 =================
OUT_DIR        = r"d:\Projects\AutoCoinAI\test"
SYMBOLS        = ["ETHUSDT","PUMPFUNUSDT"]
TIMEFRAMES     = ["15","30","60"]

# EMA 파라미터
EMA_FAST_ARR   = [3, 5, 8]
EMA_SLOW_ARR   = [9, 13, 20]

# 스토캐스틱 파라미터 (%K만 사용)
STOCH_PERIODS      = [10,14]   # %K 기준 기간
STOCH_K_SMOOTH_ARR = [3]        # %K smoothing
STOCH_D_SMOOTH_ARR = [3]        # D는 사실상 미사용이지만 파일명 파라미터 호환용

# 스토캐스틱 과매수/과매도 레벨
STOCH_OS_LEVELS = [20]          # 롱 트리거: 20을 "찍은 뒤"
STOCH_OB_LEVELS = [80]          # 숏 트리거: 80을 "찍은 뒤"

# 어느 사이드를 거래할지
SIDES          = ["BOTH"]       # "BOTH" | "LONG_ONLY" | "SHORT_ONLY"

# TP/SL (ROE%) — 레버리지 반영 수익률 기준
TP_ROE_ARR     = [5, 7.5, 10, 15]
SL_ROE_ARR     = [5, 7.5, 10]

# 자금/레버리지
EQUITY         = 100.0
LEVERAGE       = 5

# 기간/호출
START          = "2025-01-01"
END            = None
MAX_CANDLES    = 20000
SLEEP_PER_REQ  = 0.15
MAX_RETRY      = 3

# 슬리피지/수수료(테이커)
SLIPPAGE_RATE   = 0.0005   # 0.05% 불리한 체결 가정
TAKER_FEE_RATE  = 0.0006   # 0.06%/side (왕복 0.12%)

# ================= Bybit HTTP =================
session = HTTP()  # 필요 시 key/secret 환경 설정

# ================= 유틸 =================
def parse_date(s: Optional[str]) -> Optional[int]:
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
        "1":"1","3":"3","5":"5","15":"15","30":"30","60":"60",
        "120":"120","240":"240","360":"360","720":"720",
        "D":"D","W":"W","M":"M"
    }
    if tf not in mapping:
        raise ValueError(f"unsupported tf: {tf}")
    return mapping[tf]

def fetch_ohlcv(symbol: str, tf: str, start_ms: Optional[int], end_ms: Optional[int], cap: Optional[int]) -> pd.DataFrame:
    """Bybit linear kline 페이징 수집 (최신→과거)"""
    interval = bybit_interval(tf)
    if start_ms is None:
        start_ms = parse_date("2018-01-01")
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows: List[Tuple[int,float,float,float,float,float]] = []
    hard_cap = cap if cap is not None else 10**12
    cur_end = end_ms

    while len(rows) < hard_cap and cur_end > start_ms:
        req_limit = int(min(1000, hard_cap - len(rows)))
        resp = None; last_exc = None
        for _ in range(MAX_RETRY):
            try:
                resp = session.get_kline(
                    category="linear", symbol=symbol,
                    interval=interval, end=cur_end, limit=req_limit
                )
                break
            except Exception as e:
                last_exc = e; time.sleep(0.3)
        if resp is None:
            raise RuntimeError(f"Bybit API error: {last_exc}")
        if resp.get("retCode") != 0:
            raise RuntimeError(resp.get("retMsg","bybit error"))

        lst = resp.get("result",{}).get("list",[])
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

def compute_k(df: pd.DataFrame, k_period: int, k_smooth: int) -> pd.Series:
    """
    %K_raw = 100 * (Close - LL(k)) / (HH(k) - LL(k))
    %K = SMA(%K_raw, k_smooth)
    """
    low_min  = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k_raw = 100.0 * (df["close"] - low_min) / (np.maximum(high_max - low_min, 1e-12))
    k = k_raw.rolling(window=max(1,k_smooth)).mean()
    return k

# ================= 교차(EMA 기준) =================
def ema_cross_up(f_prev: float, f_now: float, s_prev: float, s_now: float) -> bool:
    # 골든: fast가 slow를 아래→위로 교차
    if np.isnan(f_prev) or np.isnan(f_now) or np.isnan(s_prev) or np.isnan(s_now):
        return False
    return f_prev <= s_prev and f_now > s_now

def ema_cross_down(f_prev: float, f_now: float, s_prev: float, s_now: float) -> bool:
    # 데드: fast가 slow를 위→아래로 교차
    if np.isnan(f_prev) or np.isnan(f_now) or np.isnan(s_prev) or np.isnan(s_now):
        return False
    return f_prev >= s_prev and f_now < s_now

# ================= 체결가(슬리피지), 수수료 반영 =================
def execution_price(side: str, px: float, on_entry: bool) -> float:
    """
    side: "LONG" or "SHORT"
    on_entry: True(진입) / False(청산)
    - LONG:  entry worse = +slip, exit worse = -slip
    - SHORT: entry worse = -slip, exit worse = +slip
    """
    s = SLIPPAGE_RATE
    if side == "LONG":
        return px * (1 + s) if on_entry else px * (1 - s)
    else:  # SHORT
        return px * (1 - s) if on_entry else px * (1 + s)

def realized_roe(side: str, entry_px: float, exit_px: float, qty: float, equity_used: float) -> Tuple[float,float]:
    """
    수수료(왕복 테이커)까지 반영한 실현 PnL/ROE(%) 반환
    """
    entry_notional = qty * entry_px
    exit_notional  = qty * exit_px
    fees = (entry_notional + exit_notional) * TAKER_FEE_RATE

    if side == "LONG":
        pnl = (exit_px - entry_px) * qty - fees
    else:  # SHORT
        pnl = (entry_px - exit_px) * qty - fees

    roe_pct = (pnl / equity_used) * 100.0
    return pnl, roe_pct

# ================= 백테스트 (K80후 EMA데드 숏 / K20후 EMA골든 롱) =================
def backtest(
    symbol: str, tf: str,
    fast: int, slow: int,
    st_p: int, st_k: int, st_d_dummy: int,   # st_d는 미사용
    os_level: float, ob_level: float,
    side_mode: str,
    tp_roe: float, sl_roe: float
) -> pd.DataFrame:

    assert fast < slow, "fast < slow 이어야 합니다."
    assert side_mode in ("BOTH","LONG_ONLY","SHORT_ONLY")

    start_ms = parse_date(START); end_ms = parse_date(END)
    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        raise SystemExit(f"[{symbol}@{tf}] no data")

    # ===== 지표 =====
    ohlc["ema_fast"] = ema(ohlc["close"], fast)
    ohlc["ema_slow"] = ema(ohlc["close"], slow)
    ohlc["K"] = compute_k(ohlc, st_p, st_k)

    # ===== 포지션 상태 =====
    position: Optional[str] = None
    entry_exec: Optional[float] = None
    qty: Optional[float] = None

    notional    = EQUITY * LEVERAGE
    equity_used = EQUITY

    # 레벨 터치 후 EMA 교차 대기 플래그
    wait_short_after_OB = False   # 80 찍은 뒤, EMA 데드크로스 대기
    wait_long_after_OS  = False   # 20 찍은 뒤, EMA 골든크로스 대기

    cols = ["datetime","symbol","timeframe",
            "fast","slow","st_p","st_k","st_d","OS","OB","side_mode",
            "포지션","비고","entry_px","exit_px","미실현PnL","ROE"]
    logs: List[List] = []

    for i in range(1, len(ohlc)):
        ts = int(ohlc.loc[i, "ts"]) // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px_now = float(ohlc.loc[i, "close"])

        # K 및 EMA 값
        k_now  = float(ohlc.loc[i,   "K"])
        k_prev = float(ohlc.loc[i-1, "K"])

        f_now  = float(ohlc.loc[i,   "ema_fast"])
        f_prev = float(ohlc.loc[i-1, "ema_fast"])
        s_now  = float(ohlc.loc[i,   "ema_slow"])
        s_prev = float(ohlc.loc[i-1, "ema_slow"])

        # --- 레벨 터치 감지 (플래그 세팅) ---
        if k_now >= ob_level or k_prev >= ob_level:
            wait_short_after_OB = True
        if k_now <= os_level or k_prev <= os_level:
            wait_long_after_OS = True

        # ===== 진입 =====
        if position is None:
            # 숏: 80 찍은 "후" EMA 데드크로스(빠른선이 느린선을 위->아래로)
            if side_mode in ("BOTH","SHORT_ONLY") and wait_short_after_OB:
                if ema_cross_down(f_prev, f_now, s_prev, s_now):
                    position = "SHORT"
                    entry_exec = execution_price("SHORT", px_now, on_entry=True)
                    qty = notional / max(entry_exec, 1e-12)
                    wait_short_after_OB = False  # 소진
                    continue

            # 롱: 20 찍은 "후" EMA 골든크로스(아래->위)
            if side_mode in ("BOTH","LONG_ONLY") and wait_long_after_OS:
                if ema_cross_up(f_prev, f_now, s_prev, s_now):
                    position = "LONG"
                    entry_exec = execution_price("LONG", px_now, on_entry=True)
                    qty = notional / max(entry_exec, 1e-12)
                    wait_long_after_OS = False
                    continue

        # ===== 청산 =====
        if position is not None:
            exit_exec = execution_price(position, px_now, on_entry=False)
            pnl, roe = realized_roe(position, entry_exec, exit_exec, qty, equity_used)

            # 1) TP/SL (ROE%)
            if roe >= tp_roe:
                logs.append([dt, symbol, tf, fast, slow, st_p, st_k, st_d_dummy, os_level, ob_level, side_mode,
                             "CLOSE", f"TP {position}", entry_exec, exit_exec, pnl, roe])
                position = entry_exec = qty = None
                continue
            if roe <= -sl_roe:
                logs.append([dt, symbol, tf, fast, slow, st_p, st_k, st_d_dummy, os_level, ob_level, side_mode,
                             "CLOSE", f"SL {position}", entry_exec, exit_exec, pnl, roe])
                position = entry_exec = qty = None
                continue

            # 2) 반대 극단 레벨 도달 시 청산
            if position == "LONG":
                if k_now >= ob_level:
                    logs.append([dt, symbol, tf, fast, slow, st_p, st_k, st_d_dummy, os_level, ob_level, side_mode,
                                 "CLOSE", "LV LONG(K>=OB)", entry_exec, exit_exec, pnl, roe])
                    position = entry_exec = qty = None
                    continue
            elif position == "SHORT":
                if k_now <= os_level:
                    logs.append([dt, symbol, tf, fast, slow, st_p, st_k, st_d_dummy, os_level, ob_level, side_mode,
                                 "CLOSE", "LV SHORT(K<=OS)", entry_exec, exit_exec, pnl, roe])
                    position = entry_exec = qty = None
                    continue

    return pd.DataFrame(logs, columns=cols)

# ================= 실행 =================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            for fast in EMA_FAST_ARR:
                for slow in EMA_SLOW_ARR:
                    if fast >= slow:  # fast는 slow보다 작아야
                        continue
                    for st_p in STOCH_PERIODS:
                        for st_k in STOCH_K_SMOOTH_ARR:
                            for st_d in STOCH_D_SMOOTH_ARR:  # D는 미사용, 파일명 호환용으로만 순회
                                for os_lv in STOCH_OS_LEVELS:
                                    for ob_lv in STOCH_OB_LEVELS:
                                        for side_mode in SIDES:
                                            for tp in TP_ROE_ARR:
                                                for sl in SL_ROE_ARR:
                                                    try:
                                                        df = backtest(
                                                            symbol=s, tf=tf,
                                                            fast=fast, slow=slow,
                                                            st_p=st_p, st_k=st_k, st_d_dummy=st_d,
                                                            os_level=os_lv, ob_level=ob_lv,
                                                            side_mode=side_mode,
                                                            tp_roe=tp, sl_roe=sl
                                                        )
                                                    except SystemExit as e:
                                                        print(f"[SKIP] {s}_{tf}_EMA{fast}-{slow}_K{st_p}-{st_k}-{st_d}_OS{os_lv}_OB{ob_lv}_{side_mode}_TP{tp}_SL{sl}: {e}")
                                                        continue
                                                    except Exception as e:
                                                        print(f"[ERR ] {s}_{tf}_EMA{fast}-{slow}_K{st_p}-{st_k}-{st_d}_OS{os_lv}_OB{ob_lv}_{side_mode}_TP{tp}_SL{sl}: {e}")
                                                        continue

                                                    fname = f"{s}_{tf}_K80_to_EMAdead_K20_to_EMAgolden_EMA{fast}-{slow}_K{st_p}-{st_k}-{st_d}_OS{os_lv}_OB{ob_lv}_{side_mode}_TP{tp}_SL{sl}.csv"

                                                    fpath = os.path.join(OUT_DIR, fname)
                                                    df.to_csv(fpath, index=False, encoding="utf-8-sig")
                                                    print(f"✅ 저장: {fpath}")
