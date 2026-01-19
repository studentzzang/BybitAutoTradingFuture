from dotenv import load_dotenv, find_dotenv
from pybit.unified_trading import HTTP
import os, sys
from datetime import datetime
import time
import bybit

load_dotenv(find_dotenv(), override=True)
_api_key = os.getenv("API_KEY")
_api_secret = os.getenv("API_KEY_SECRET")
if not _api_key or not _api_secret:
    print("❌ API_KEY 또는 API_KEY_SECRET을 .env에서 못 찾았습니다.")
    sys.exit(1)

session = HTTP(api_key=_api_key , api_secret=_api_secret, recv_window=10000, max_retries=0)

# ===================== 사용자 설정 =====================

SYMBOLS      = ["PUMPFUNUSDT"]
RSI_PERIODS  = [9]
INTERVALS    = ["30"]

LONG_SWITCH_RSI  = [28]  # 과매도 기준 (롱 방향)
SHORT_SWITCH_RSI = [72]  # 과매수 기준 (숏 방향)

LEVERAGE      = "5"
PCT           = 40
COOLDOWN_BARS = 0

DOORSTEP      = 2.0      # 진입: RSI 피크/바닥에서 이만큼 이동 후 진입

TP_ROE  = [15]           # TP 기준(ROE %)
SL_ROE  = [10]           # SL 기준(ROE %)
TP_MODE = [2]            # 1: 모드1 (RSI 반대 과상태 + doorstep 트레일링), 2: 그냥 TP/SL

# ===================== 상태 변수 =====================

position      = {s: None for s in SYMBOLS}   # "long" / "short" / None
entry_px      = {s: None for s in SYMBOLS}
init_margin   = {s: None for s in SYMBOLS}
qty           = {s: None for s in SYMBOLS}

# RSI 스위치용 (doorstep 진입 로직)
last_peak_level    = {s: None for s in SYMBOLS}  # 숏용 RSI 피크
last_trough_level  = {s: None for s in SYMBOLS}  # 롱용 RSI 바닥
armed_short_switch = {s: False for s in SYMBOLS}
armed_long_switch  = {s: False for s in SYMBOLS}

# 봉 기준 쿨다운
last_closed_price1 = {s: None for s in SYMBOLS}
cooldown_bars      = {s: 0   for s in SYMBOLS}

# 모드1 TP 유지용
tp_hold   = {s: False for s in SYMBOLS}   # TP 돌파 후 "유지 모드"인지
roe_peak  = {s: None  for s in SYMBOLS}   # TP 이후 ROE 최고값

# bybit 모듈 설정
bybit.PCT = PCT
for s in SYMBOLS:
    bybit.SYMBOLS.append(s)

BASE_CASH = None


# ===================== 유틸 함수 =====================

def start():
    """시작 시 USDT 잔고 및 레버리지 설정"""
    global BASE_CASH
    BASE_CASH = bybit.get_usdt()
    print(f"🔧 보유금액: {BASE_CASH:.2f} USDT")
    for s in SYMBOLS:
        bybit.set_leverage(symbol=s, leverage=LEVERAGE)


def reset_switch_after_close(symbol, closed_side):
    """포지션 청산 후 RSI 스위치 상태 리셋"""
    if closed_side == "long":
        # 롱 끝났으면 다음에 롱 다시 잡을 수 있게 롱 스위치만 켜두고 바닥값 리셋
        armed_long_switch[symbol] = True
        last_trough_level[symbol] = None
    elif closed_side == "short":
        armed_short_switch[symbol] = True
        last_peak_level[symbol] = None

    # 모드1 상태도 같이 리셋
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


def close_long(symbol):
    """롱 포지션 청산 (Sell) + 상태 리셋 일부 공통 처리용"""
    bybit.close_position(symbol, "Sell")


def close_short(symbol):
    """숏 포지션 청산 (Buy) + 상태 리셋 일부 공통 처리용"""
    bybit.close_position(symbol, "Buy")


def enter_long(symbol, px, q, leverage):
    position[symbol]    = "long"
    entry_px[symbol]    = px
    qty[symbol]         = q
    init_margin[symbol] = (px * q) / float(leverage)
    # TP 모드1 상태 리셋
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


def enter_short(symbol, px, q, leverage):
    position[symbol]    = "short"
    entry_px[symbol]    = px
    qty[symbol]         = q
    init_margin[symbol] = (px * q) / float(leverage)
    # TP 모드1 상태 리셋
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


# ===================== 메인 루프 =====================

def update():
    while True:
        for idx, symbol in enumerate(SYMBOLS):
            try:
                tp_roe  = TP_ROE[idx]
                sl_roe  = SL_ROE[idx]
                tp_mode = TP_MODE[idx]

                rsi_period = RSI_PERIODS[idx]
                interval   = INTERVALS[idx]
                long_rsi   = LONG_SWITCH_RSI[idx]
                short_rsi  = SHORT_SWITCH_RSI[idx]

                # 현재 PnL, ROE, 가격, RSI
                Pnl = bybit.get_PnL(symbol)
                ROE = bybit.get_ROE(symbol)   # 여기 ROE를 기준으로 TP/SL/doorstep 트레일링
                c_prev2, c_prev1, cur_3 = bybit.get_close_price(symbol, interval=interval)
                RSI = bybit.get_RSI(symbol, interval=interval, period=rsi_period)

                # 새 봉 체크 (쿨다운용)
                new_bar = (last_closed_price1[symbol] is None) or (last_closed_price1[symbol] != c_prev1)
                if new_bar:
                    last_closed_price1[symbol] = c_prev1
                    if cooldown_bars[symbol] > 0:
                        cooldown_bars[symbol] -= 1

                # ===== 1) RSI 스위치 업데이트 (doorstep 진입용) =====

                # 롱 방향: RSI가 long_rsi 이하로 내려갔을 때
                if RSI <= long_rsi:
                    if not armed_long_switch[symbol]:
                        armed_long_switch[symbol] = True
                        last_trough_level[symbol] = RSI
                    else:
                        if last_trough_level[symbol] is None or RSI < last_trough_level[symbol]:
                            last_trough_level[symbol] = RSI

                # 숏 방향: RSI가 short_rsi 이상으로 올라갔을 때
                if RSI >= short_rsi:
                    if not armed_short_switch[symbol]:
                        armed_short_switch[symbol] = True
                        last_peak_level[symbol] = RSI
                    else:
                        if last_peak_level[symbol] is None or RSI > last_peak_level[symbol]:
                            last_peak_level[symbol] = RSI

                # ===== 2) 포지션 없음 & 쿨다운 끝 → 진입 =====
                if position[symbol] is None and cooldown_bars[symbol] == 0:

                    # (1) 숏 진입: RSI 피크 찍고 DOORSTEP만큼 내려왔을 때
                    if armed_short_switch[symbol] and last_peak_level[symbol] is not None:
                        short_trigger = last_peak_level[symbol] - DOORSTEP
                        if RSI <= short_trigger:
                            px, q = bybit.entry_position(symbol, "Sell", LEVERAGE)
                            if q > 0 and px is not None:
                                enter_short(symbol, px, q, LEVERAGE)
                                armed_short_switch[symbol] = False
                                last_peak_level[symbol] = None
                                cooldown_bars[symbol] = COOLDOWN_BARS

                    # (2) 롱 진입: RSI 바닥 찍고 DOORSTEP만큼 올라왔을 때
                    if position[symbol] is None and cooldown_bars[symbol] == 0:
                        if armed_long_switch[symbol] and last_trough_level[symbol] is not None:
                            long_trigger = last_trough_level[symbol] + DOORSTEP
                            if RSI >= long_trigger:
                                px, q = bybit.entry_position(symbol, "Buy", LEVERAGE)
                                if q > 0 and px is not None:
                                    enter_long(symbol, px, q, LEVERAGE)
                                    armed_long_switch[symbol] = False
                                    last_trough_level[symbol] = None
                                    cooldown_bars[symbol] = COOLDOWN_BARS

                # ===== 3) 포지션 보유 시 청산 로직 =====
                closed = False
                closed_side = None

                # ---- 숏 포지션 ----
                if position[symbol] == "short":
                    roe = ROE  # bybit에서 받은 ROE 그대로 사용

                    # (a) SL 먼저 체크
                    if roe <= -sl_roe:
                        close_short(symbol)
                        closed = True
                        closed_side = "short"

                    # (b) TP MODE 처리
                    if not closed:
                        if tp_mode == 1:
                            # 모드1: TP 돌파 후 RSI 반대 과상태일 때 버티다가,
                            # ROE가 피크에서 DOORSTEP만큼 떨어지면 익절.

                            # 1) 아직 hold 모드 아니고, TP 처음 돌파
                            if not tp_hold[symbol] and roe >= tp_roe:
                                tp_hold[symbol]  = True
                                roe_peak[symbol] = roe

                            # 2) hold 모드일 때
                            if tp_hold[symbol]:
                                # 숏이니까 반대 과상태 = 과매도 → RSI <= long_rsi
                                if RSI <= long_rsi:
                                    # ROE 피크 갱신
                                    if roe > roe_peak[symbol]:
                                        roe_peak[symbol] = roe

                                    # 피크에서 DOORSTEP만큼 하락하면 청산
                                    if roe_peak[symbol] - roe >= DOORSTEP:
                                        close_short(symbol)
                                        closed = True
                                        closed_side = "short"
                                else:
                                    # 반대 과상태 벗어나면 그냥 TP 익절
                                    close_short(symbol)
                                    closed = True
                                    closed_side = "short"
                        else:
                            # 기본 모드: TP / SL 단순 조건
                            if roe >= tp_roe or roe <= -sl_roe:
                                close_short(symbol)
                                closed = True
                                closed_side = "short"

                # ---- 롱 포지션 ----
                elif position[symbol] == "long":
                    roe = ROE

                    # (a) SL 먼저 체크
                    if roe <= -sl_roe:
                        close_long(symbol)
                        closed = True
                        closed_side = "long"

                    # (b) TP MODE 처리
                    if not closed:
                        if tp_mode == 1:
                            # 롱: TP 돌파 후 과매수(숏 방향) RSI 상태 유지하며 버티다가
                            # ROE가 피크에서 DOORSTEP만큼 떨어지면 익절.

                            # 1) 아직 hold 모드 아니고 TP 처음 돌파
                            if not tp_hold[symbol] and roe >= tp_roe:
                                tp_hold[symbol]  = True
                                roe_peak[symbol] = roe

                            # 2) hold 모드일 때
                            if tp_hold[symbol]:
                                # 롱이니까 반대 과상태 = 과매수 → RSI >= short_rsi
                                if RSI >= short_rsi:
                                    if roe > roe_peak[symbol]:
                                        roe_peak[symbol] = roe

                                    if roe_peak[symbol] - roe >= DOORSTEP:
                                        close_long(symbol)
                                        closed = True
                                        closed_side = "long"
                                else:
                                    # 반대 과상태 벗어나면 그냥 TP 익절
                                    close_long(symbol)
                                    closed = True
                                    closed_side = "long"
                        else:
                            # 기본 모드: TP / SL 단순
                            if roe >= tp_roe or roe <= -sl_roe:
                                close_long(symbol)
                                closed = True
                                closed_side = "long"

                # ---- 청산 후 공통 처리 ----
                if closed:
                    position[symbol]    = None
                    entry_px[symbol]    = None
                    qty[symbol]         = None
                    init_margin[symbol] = None
                    cooldown_bars[symbol] = COOLDOWN_BARS
                    reset_switch_after_close(symbol, closed_side)

                # ===== 4) 상태 출력 (이모지 그대로 유지) =====
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"🪙 {symbol} 🕧 {interval} | 🚩포지션:{position[symbol]} "
                    f"| RSI:{RSI:.2f} |💸 PnL:{Pnl:.3f} |💎 ROE:{ROE:.2f} "
                )

            except Exception as e:
                print(f"[ERROR] {symbol}: {type(e).__name__} {e}")
                continue

            time.sleep(5)
        time.sleep(3)


start()
update()
