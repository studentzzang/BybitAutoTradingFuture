import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# =========================
# Stochastic 백테스트 CSV 메이커
# - 진입: Stoch K/D 크로스 + (과매도/과매수 구간에서 발생한 크로스만)
# - 모드1: TP/SL(ROE%)만으로 청산 (크로스 청산 X)
# - 모드2: 반대 크로스 나오면 ROE 상관없이 청산 (TP/SL 없음, 파일명에도 안 넣음)
# - 파라미터는 전부 배열로 돌리고, 조합별 CSV 생성
# - 체결은 "신호봉 close 확인 -> 다음봉 open 체결" (룩어헤드 최소화)
# =========================

# ====== 사용자 설정 ======
SYMBOLS = ["PUMPFUNUSDT","ETHUSDT","XRPUSDT"]
LEVERAGE_ARR = [10]
TIMEFRAMES = [3, 5, 15,30]          # 분봉
EQUITY_ARR = [100.0]            # USDT 기준 (단순 가정)
START = "2025-09-01"
END   = "2026-01-05" 
OUT_DIR = "test"
MAX_CANDLES = 50000

# ====== Stochastic 설정 ======
STOCH_PERIOD_ARR = [14]
K_SMOOTH_ARR = [5,3]
D_SMOOTH_ARR = [3]

# ====== 과매도/과매수 구간 ======
OVERSOLD_ARR = [20.0]
OVERBOUGHT_ARR = [80.0]

# "구간에서 크로스" 판정 방식
# True: 신호봉에서 K와 D가 둘 다 구간 안이어야 함(엄격)
# False: 신호봉에서 K만 구간 안이면 OK(느슨)
STRICT_ZONE_ARR = [True, False]

# ====== 모드 ======
MODES = [1, 2]

# 모드1 전용 TP/SL (ROE% 기준)
TP_ROE_ARR = [3.0, 6, 10]
SL_ROE_ARR = [3.0,6,10]

# ====== (선택) 수수료/슬리피지 ======
FEE_RATE = 0.0        # 예: 0.00055
SLIPPAGE = 0.0        # 예: 0.0001

# ====== Bybit 세션 (Public OHLCV) ======
session = HTTP(testnet=False)

# ----------------- 유틸 -----------------
def parse_date_ms(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def apply_costs(entry_px: float, exit_px: float, qty: float) -> float:
    # 왕복 수수료 + 슬리피지(단순)
    notional_entry = entry_px * qty
    notional_exit  = exit_px * qty
    fee = (notional_entry + notional_exit) * FEE_RATE
    slip = (notional_entry + notional_exit) * SLIPPAGE
    return fee + slip

def fetch_ohlcv(symbol: str, tf_min: int, start_ms: int, end_ms: int, limit: int) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    interval = str(tf_min)

    while True:
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=cursor,
            end=end_ms,
            limit=1000,
        )
        data = resp.get("result", {}).get("list", [])
        if not data:
            break

        data = sorted(data, key=lambda x: int(x[0]))
        for k in data:
            ts = int(k[0])
            if ts < start_ms or ts > end_ms:
                continue
            rows.append([ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])

        last_ts = int(data[-1][0])
        if last_ts <= cursor:
            break
        cursor = last_ts + tf_min * 60 * 1000

        if len(rows) >= limit:
            break
        time.sleep(0.02)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.sort_values("ts", inplace=True)
    df.drop_duplicates("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def compute_stoch(df: pd.DataFrame, period: int, k_smooth: int, d_smooth: int) -> pd.DataFrame:
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)

    k_raw = 100.0 * (df["close"] - low_min) / denom
    k = k_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()

    out = df.copy()
    out["k"] = k
    out["d"] = d
    return out

def calc_pnl_roe(position: str, entry_px: float, exit_px: float, qty: float, equity_used: float) -> tuple[float, float]:
    pnl = (exit_px - entry_px) * qty if position == "LONG" else (entry_px - exit_px) * qty
    roe = (pnl / equity_used) * 100.0 if equity_used > 0 else 0.0
    return pnl, roe

def in_zone(k: float, d: float, zone: float, strict: bool, is_oversold: bool) -> bool:
    # oversold zone: <= zone
    # overbought zone: >= zone
    if is_oversold:
        return (k <= zone and d <= zone) if strict else (k <= zone)
    else:
        return (k >= zone and d >= zone) if strict else (k >= zone)

# ----------------- 백테스트 -----------------
def backtest_stoch(
    symbol: str,
    tf: int,
    leverage: int,
    equity: float,
    period: int,
    k_smooth: int,
    d_smooth: int,
    oversold: float,
    overbought: float,
    strict_zone: bool,
    mode: int,
    tp_roe: Optional[float] = None,
    sl_roe: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:

    start_ms = parse_date_ms(START)
    end_ms = parse_date_ms(END) + 24 * 60 * 60 * 1000 - 1

    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty or len(ohlc) < 200:
        return pd.DataFrame(), {"trades": 0}

    ohlc = compute_stoch(ohlc, period, k_smooth, d_smooth)
    ohlc.dropna(inplace=True)
    ohlc.reset_index(drop=True, inplace=True)

    # 실행가: 다음 봉 open
    opens = ohlc["open"].to_numpy()
    closes = ohlc["close"].to_numpy()
    ks = ohlc["k"].to_numpy()
    ds = ohlc["d"].to_numpy()
    ts = ohlc["ts"].to_numpy()

    position: Optional[str] = None
    entry_px = 0.0
    entry_i = -1
    qty = 0.0

    equity_used = equity
    notional = equity * leverage

    logs = []

    wins = 0
    total_pnl = 0.0

    # i는 "신호 확인 봉" 인덱스, 체결은 i+1 open
    for i in range(1, len(ohlc) - 1):
        k_prev, d_prev = float(ks[i - 1]), float(ds[i - 1])
        k_now, d_now   = float(ks[i]), float(ds[i])

        # 신호봉 종료 후 다음봉 open에서 체결
        exec_px = float(opens[i + 1])
        exec_dt = ms_to_dt(int(ts[i + 1]))

        bull_cross = (k_prev < d_prev) and (k_now > d_now)  # 골든
        bear_cross = (k_prev > d_prev) and (k_now < d_now)  # 데드

        # ====== 포지션 없으면 진입 ======
        if position is None:
            # 롱: 과매도 구간에서 bullish cross
            if bull_cross and in_zone(k_now, d_now, oversold, strict_zone, is_oversold=True):
                position = "LONG"
                entry_px = exec_px
                entry_i = i + 1
                qty = notional / entry_px
                logs.append([exec_dt, symbol, tf, mode, "OPEN_LONG", entry_px, entry_px, 0.0, 0.0, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone])
                continue

            # 숏: 과매수 구간에서 bearish cross
            if bear_cross and in_zone(k_now, d_now, overbought, strict_zone, is_oversold=False):
                position = "SHORT"
                entry_px = exec_px
                entry_i = i + 1
                qty = notional / entry_px
                logs.append([exec_dt, symbol, tf, mode, "OPEN_SHORT", entry_px, entry_px, 0.0, 0.0, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone])
                continue

        # ====== 포지션 있으면 청산 조건 ======
        else:
            # 현재 시점 평가도 exec_px(i+1 open) 기준으로 처리 (룩어헤드 줄이기)
            pnl, roe = calc_pnl_roe(position, entry_px, exec_px, qty, equity_used)

            # 모드1: TP/SL만
            if mode == 1:
                if tp_roe is None or sl_roe is None:
                    raise ValueError("mode 1 needs tp_roe and sl_roe")

                # 청산
                if roe >= tp_roe or roe <= -sl_roe:
                    cost = apply_costs(entry_px, exec_px, qty)
                    pnl_net = pnl - cost
                    roe_net = (pnl_net / equity_used) * 100.0 if equity_used > 0 else 0.0

                    action = "CLOSE_TP" if roe >= tp_roe else "CLOSE_SL"
                    logs.append([exec_dt, symbol, tf, mode, action, entry_px, exec_px, pnl_net, roe_net, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone])

                    total_pnl += pnl_net
                    if pnl_net > 0:
                        wins += 1

                    position = None
                    entry_px = 0.0
                    entry_i = -1
                    qty = 0.0
                continue

            # 모드2: 반대 크로스 나오면 ROE 상관없이 청산
            if mode == 2:
                exit_now = False
                # 롱이면 bearish cross가 "반대 크로스"
                if position == "LONG" and bear_cross:
                    exit_now = True
                # 숏이면 bullish cross가 "반대 크로스"
                if position == "SHORT" and bull_cross:
                    exit_now = True

                if exit_now:
                    cost = apply_costs(entry_px, exec_px, qty)
                    pnl_net = pnl - cost
                    roe_net = (pnl_net / equity_used) * 100.0 if equity_used > 0 else 0.0
                    logs.append([exec_dt, symbol, tf, mode, "CLOSE_X", entry_px, exec_px, pnl_net, roe_net, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone])

                    total_pnl += pnl_net
                    if pnl_net > 0:
                        wins += 1

                    position = None
                    entry_px = 0.0
                    entry_i = -1
                    qty = 0.0
                continue

    df = pd.DataFrame(
        logs,
        columns=[
            "datetime", "symbol", "tf", "mode", "action",
            "entry_px", "exec_px", "pnl", "roe",
            "k", "d", "oversold", "overbought", "strict_zone",
        ],
    )

    # 요약
    closes = df[df["action"].str.startswith("CLOSE")]
    trades = len(closes)
    winrate = (wins / trades) * 100.0 if trades > 0 else 0.0

    summary = {
        "symbol": symbol,
        "tf": tf,
        "mode": mode,
        "leverage": leverage,
        "equity": equity,
        "period": period,
        "k_smooth": k_smooth,
        "d_smooth": d_smooth,
        "oversold": oversold,
        "overbought": overbought,
        "strict_zone": strict_zone,
        "tp_roe": tp_roe if mode == 1 else "",
        "sl_roe": sl_roe if mode == 1 else "",
        "trades": trades,
        "winrate_pct": winrate,
        "total_pnl": total_pnl,
    }

    return df, summary

# ----------------- 실행 -----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = []

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            for lev in LEVERAGE_ARR:
                for eq in EQUITY_ARR:
                    for period in STOCH_PERIOD_ARR:
                        for ks in K_SMOOTH_ARR:
                            for ds in D_SMOOTH_ARR:
                                for os_ in OVERSOLD_ARR:
                                    for ob_ in OVERBOUGHT_ARR:
                                        for strict in STRICT_ZONE_ARR:
                                            for mode in MODES:
                                                # 모드2는 tp/sl 없음
                                                if mode == 2:
                                                    df, summ = backtest_stoch(
                                                        symbol, tf, lev, eq,
                                                        period, ks, ds,
                                                        os_, ob_, strict,
                                                        mode=2,
                                                        tp_roe=None, sl_roe=None
                                                    )
                                                    if df.empty:
                                                        continue

                                                    fname = (
                                                        f"{symbol}_tf{tf}_L{lev}"
                                                        f"_STOCHp{period}_k{ks}_d{ds}"
                                                        f"_OS{os_}_OB{ob_}_strict{strict}"
                                                        f"_MODE2.csv"
                                                    )
                                                    df.to_csv(os.path.join(OUT_DIR, fname), index=False, encoding="utf-8-sig")
                                                    print("✅ Saved:", fname)
                                                    summaries.append(summ)
                                                    continue

                                                # 모드1은 tp/sl 조합까지
                                                for tp in TP_ROE_ARR:
                                                    for sl in SL_ROE_ARR:
                                                        df, summ = backtest_stoch(
                                                            symbol, tf, lev, eq,
                                                            period, ks, ds,
                                                            os_, ob_, strict,
                                                            mode=1,
                                                            tp_roe=tp, sl_roe=sl
                                                        )
                                                        if df.empty:
                                                            continue

                                                        fname = (
                                                            f"{symbol}_tf{tf}_L{lev}"
                                                            f"_STOCHp{period}_k{ks}_d{ds}"
                                                            f"_OS{os_}_OB{ob_}_strict{strict}"
                                                            f"_TP{tp}_SL{sl}"
                                                            f"_MODE1.csv"
                                                        )
                                                        df.to_csv(os.path.join(OUT_DIR, fname), index=False, encoding="utf-8-sig")
                                                        print("✅ Saved:", fname)
                                                        summaries.append(summ)

    # summary 저장
    if summaries:
        sdfs = pd.DataFrame(summaries)
        sdfs.to_csv(os.path.join(OUT_DIR, "SUMMARY.csv"), index=False, encoding="utf-8-sig")
        print("✅ Saved: SUMMARY.csv")
    else:
        print("⚠️ No results. (데이터 부족이거나 조건이 너무 빡셈)")

if __name__ == "__main__":
    main()
