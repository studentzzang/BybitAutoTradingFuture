import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# =========================
# Stochastic 백테스트 CSV 메이커 (RSI.py 스타일 복리 누적 + TP/SL 고정 반영)
# - 진입: Stoch K/D 크로스 + (strict_zone면 과매도/과매수 구간에서 발생한 크로스만)
# - 모드1: TP/SL(ROE%)만으로 청산
#          * TP/SL 트리거는 가격변화로 판단하지만, 기록/누적은 '고정 ROE'로 처리 (분봉 커도 들쭉날쭉 방지)
# - 모드2: 반대 크로스 나오면 ROE 상관없이 청산 (고정 TP/SL 적용 X)
# - 체결: '신호봉 close 확인 -> 다음봉 open 체결' (룩어헤드 최소화)
# - CSV: equity_before/equity_after 포함 (증거금 누적 확인 가능)
# =========================

# ====== 사용자 설정 ======
SYMBOLS = ["PUMPFUNUSDT"]
TF_ARR = [5,15,30,60]
MODE_ARR = [1,2]                 # 1=TP/SL, 2=반대크로스청산
LEVERAGE_ARR = [5]
EQUITY_ARR = [100.0]

# 날짜 범위(UTC 기준 YYYY-MM-DD)
START = "2026-01-12"
END   = "2026-02-12"

# 캔들 최대 수집
MAX_CANDLES = 40000

# 스토캐스틱 파라미터
PERIOD_ARR = [14,9,5]
K_SMOOTH_ARR = [5,3]
D_SMOOTH_ARR = [5,3]

# 존 파라미터
OVERSOLD_ARR = [20.0,25]
OVERBOUGHT_ARR = [80.0,75]
STRICT_ZONE_ARR = [False]

# TP/SL (ROE, %)
TP_ROE_ARR = [5,10,15]
SL_ROE_ARR = [4,7]

# 비용(단순)
TAKER_FEE = 0
SLIPPAGE = 0

# 결과 저장 폴더
OUT_DIR = "test"


# ====== 시간 유틸 ======
def parse_date_ms(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ====== 비용 ======
def apply_costs(entry_px: float, exit_px: float, qty: float) -> float:
    # 양방향(진입+청산) notional 기준으로 대충 비용 적용
    notional_in = abs(qty) * entry_px
    notional_out = abs(qty) * exit_px
    fee = (notional_in + notional_out) * TAKER_FEE
    slip = (notional_in + notional_out) * SLIPPAGE
    return fee + slip


# ====== Stochastic ======
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


def in_zone(k: float, d: float, threshold: float, strict_zone: bool, is_oversold: bool) -> bool:
    # strict_zone=True면 K,D 둘 다 threshold 안쪽에 있어야 함
    if not strict_zone:
        return True
    if is_oversold:
        return (k <= threshold) and (d <= threshold)
    else:
        return (k >= threshold) and (d >= threshold)


def calc_pnl_roe(position: str, entry_px: float, exit_px: float, qty: float, equity_used: float) -> Tuple[float, float]:
    # 선물 단순 PnL (USDT) + ROE(%)
    if position == "LONG":
        pnl = (exit_px - entry_px) * qty
    else:
        pnl = (entry_px - exit_px) * qty
    roe = (pnl / equity_used) * 100.0 if equity_used > 0 else 0.0
    return pnl, roe


# ====== Bybit OHLCV (RSI.py와 동일: end를 줄여가며 과거로 paging) ======
session = HTTP(testnet=False)

def fetch_ohlcv(symbol: str, tf_min: int, start_ms: int, end_ms: int, max_candles: int) -> pd.DataFrame:
    interval = str(tf_min)
    rows = []
    cur_end = int(end_ms)

    while len(rows) < max_candles:
        resp = None
        for _ in range(3):
            try:
                r = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    end=cur_end,
                    limit=1000
                )
                if isinstance(r, dict) and r.get("retCode") == 0:
                    resp = r
                    break
            except Exception:
                pass
            time.sleep(0.4)

        if resp is None:
            break

        lst = resp.get("result", {}).get("list", [])
        if not lst:
            break

        batch_min_ts = None
        for it in lst:
            ts = int(it[0])
            o, h, l, c, v = map(float, it[1:6])
            rows.append((ts, o, h, l, c, v))
            if batch_min_ts is None or ts < batch_min_ts:
                batch_min_ts = ts

        cur_end = min(int(x[0]) for x in lst) - 1

        if batch_min_ts is not None and batch_min_ts <= start_ms:
            break
        if len(lst) < 1000:
            break
        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[(df["ts"] >= start_ms) & (df["ts"] <= end_ms)].copy()
    df.reset_index(drop=True, inplace=True)
    if not df.empty:
        print(f"[FETCH] {symbol} {tf_min}m candles={len(df)} range={ms_to_dt(int(df['ts'].iloc[0]))} ~ {ms_to_dt(int(df['ts'].iloc[-1]))}")
    else:
        print(f"[FETCH] {symbol} {tf_min}m candles=0 (no data in range)")
    return df


# ====== 백테스트 ======
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
) -> Tuple[pd.DataFrame, Dict]:

    start_ms = parse_date_ms(START)
    end_ms = parse_date_ms(END) + 24 * 60 * 60 * 1000 - 1

    ohlc = fetch_ohlcv(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty or len(ohlc) < 200:
        return pd.DataFrame(), {"trades": 0}

    ohlc = compute_stoch(ohlc, period, k_smooth, d_smooth)
    ohlc.dropna(inplace=True)
    ohlc.reset_index(drop=True, inplace=True)

    opens = ohlc["open"].to_numpy()
    highs = ohlc["high"].to_numpy()
    lows  = ohlc["low"].to_numpy()
    ks = ohlc["k"].to_numpy()
    ds = ohlc["d"].to_numpy()
    ts = ohlc["ts"].to_numpy()

    position: Optional[str] = None
    entry_px = 0.0
    qty = 0.0
    entry_equity = equity  # 진입 당시 증거금(고정 pnl/roe 계산 기준)

    # ✅ 복리: 현재 증거금
    current_equity = float(equity)

    logs = []
    header = ["datetime","symbol","tf","mode","action","entry_px","exec_px","pnl","roe","k","d","oversold","overbought","strict_zone","equity_before","equity_after"]

    total_pnl = 0.0
    wins = 0
    closes = 0

    for i in range(1, len(ohlc) - 1):
        k_prev, d_prev = float(ks[i - 1]), float(ds[i - 1])
        k_now, d_now   = float(ks[i]), float(ds[i])

        # 신호봉 종료 후 다음봉 open에서 체결
        exec_px = float(opens[i + 1])
        exec_dt = ms_to_dt(int(ts[i + 1]))

        bull_cross = (k_prev < d_prev) and (k_now > d_now)
        bear_cross = (k_prev > d_prev) and (k_now < d_now)

        # ====== 포지션 없으면 진입 ======
        if position is None:
            notional = current_equity * leverage  # ✅ 복리 반영
            if bull_cross and in_zone(k_now, d_now, oversold, strict_zone, is_oversold=True):
                position = "LONG"
                entry_px = exec_px
                qty = notional / entry_px
                entry_equity = current_equity
                logs.append([exec_dt, symbol, tf, mode, "OPEN_LONG", entry_px, entry_px, 0.0, 0.0, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone, current_equity, current_equity])
                continue

            if bear_cross and in_zone(k_now, d_now, overbought, strict_zone, is_oversold=False):
                position = "SHORT"
                entry_px = exec_px
                qty = notional / entry_px
                entry_equity = current_equity
                logs.append([exec_dt, symbol, tf, mode, "OPEN_SHORT", entry_px, entry_px, 0.0, 0.0, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone, current_equity, current_equity])
                continue

            continue

        # ====== 포지션 있으면 청산 ======
        pnl, roe = calc_pnl_roe(position, entry_px, exec_px, qty, entry_equity)

        if mode == 1:
            if tp_roe is None or sl_roe is None:
                raise ValueError("mode 1 needs tp_roe and sl_roe")

            # TP/SL 트리거는 여전히 ROE 기준이지만,
            # 누적/기록은 고정(+tp_roe / -sl_roe)로 처리
            if roe >= tp_roe or roe <= -sl_roe:
                equity_before = current_equity

                is_tp = roe >= tp_roe
                fixed_roe = float(tp_roe) if is_tp else -float(sl_roe)
                fixed_pnl = entry_equity * (fixed_roe / 100.0)  # ✅ RSI.py 스타일: 진입 당시 증거금 기준 고정

                cost = apply_costs(entry_px, exec_px, qty)
                pnl_net = fixed_pnl - cost

                current_equity = equity_before + pnl_net
                roe_net = (pnl_net / equity_before) * 100.0 if equity_before > 0 else 0.0

                action = "CLOSE_TP" if is_tp else "CLOSE_SL"
                logs.append([exec_dt, symbol, tf, mode, action, entry_px, exec_px, pnl_net, roe_net, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone, equity_before, current_equity])

                total_pnl += pnl_net
                closes += 1
                if pnl_net > 0:
                    wins += 1

                position = None
                entry_px = 0.0
                qty = 0.0
                entry_equity = current_equity
                continue

            continue

        if mode == 2:
            # 반대 크로스면 즉시 청산(고정 TP/SL 없음)
            close_now = (position == "LONG" and bear_cross) or (position == "SHORT" and bull_cross)
            if close_now:
                equity_before = current_equity
                cost = apply_costs(entry_px, exec_px, qty)
                pnl_net = pnl - cost
                current_equity = equity_before + pnl_net
                roe_net = (pnl_net / equity_before) * 100.0 if equity_before > 0 else 0.0
                logs.append([exec_dt, symbol, tf, mode, "CLOSE_X", entry_px, exec_px, pnl_net, roe_net, float(ks[i]), float(ds[i]), oversold, overbought, strict_zone, equity_before, current_equity])

                total_pnl += pnl_net
                closes += 1
                if pnl_net > 0:
                    wins += 1

                position = None
                entry_px = 0.0
                qty = 0.0
                entry_equity = current_equity
            continue

    df = pd.DataFrame(logs, columns=header)
    stats = {
        "symbol": symbol,
        "tf": tf,
        "mode": mode,

        # ===== Stochastic params =====
        "stoch_period": period,
        "k_smooth": k_smooth,
        "d_smooth": d_smooth,
        "oversold": oversold,
        "overbought": overbought,
        "strict_zone": strict_zone,

        # ===== Exit params =====
        # mode 1: uses tp_roe/sl_roe
        # mode 2: will remain None (blank in SUMMARY)
        "tp_roe": tp_roe,
        "sl_roe": sl_roe,

        # ===== Capital params =====
        "leverage": leverage,
        "equity_start": equity,

        "trades": closes,
        "wins": wins,
        "winrate": (wins / closes * 100.0) if closes else 0.0,
        "final_equity": float(df["equity_after"].iloc[-1]) if len(df) else equity,
        "total_pnl": total_pnl,
    }
    return df, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = []
    for symbol in SYMBOLS:
        for tf in TF_ARR:
            for mode in MODE_ARR:
                for lev in LEVERAGE_ARR:
                    for eq in EQUITY_ARR:
                        for p in PERIOD_ARR:
                            for ks in K_SMOOTH_ARR:
                                for ds in D_SMOOTH_ARR:
                                    for osd in OVERSOLD_ARR:
                                        for obd in OVERBOUGHT_ARR:
                                            for strict in STRICT_ZONE_ARR:
                                                if mode == 1:
                                                    for tp in TP_ROE_ARR:
                                                        for sl in SL_ROE_ARR:
                                                            df, st = backtest_stoch(symbol, tf, lev, eq, p, ks, ds, osd, obd, strict, mode, tp, sl)
                                                            if df.empty:
                                                                continue
                                                            fname = f"{symbol}_{tf}_STOCH_P{p}_TP{tp}_SL{sl}_M{mode}.csv"
                                                            path = os.path.join(OUT_DIR, fname)
                                                            df.to_csv(path, index=False, encoding="utf-8-sig")
                                                            st["csv"] = fname
                                                            summary.append(st)
                                                else:
                                                    df, st = backtest_stoch(symbol, tf, lev, eq, p, ks, ds, osd, obd, strict, mode)
                                                    if df.empty:
                                                        continue
                                                    fname = f"{symbol}_{tf}_STOCH_P{p}_M{mode}.csv"
                                                    path = os.path.join(OUT_DIR, fname)
                                                    df.to_csv(path, index=False, encoding="utf-8-sig")
                                                    st["csv"] = fname
                                                    summary.append(st)

    if summary:
        s_df = pd.DataFrame(summary)
        s_df.sort_values("final_equity", ascending=False, inplace=True)
        s_df.to_csv(os.path.join(OUT_DIR, "SUMMARY.csv"), index=False, encoding="utf-8-sig")
        print("✅ SUMMARY 저장 완료")


if __name__ == "__main__":
    main()
