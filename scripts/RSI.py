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
    print(f"cwd={os.getcwd()}  .env={find_dotenv() or 'NOT FOUND'}")
    sys.exit(1)

session = HTTP(api_key=_api_key, api_secret=_api_secret, recv_window=10000, max_retries=0)

SYMBOLS = ["PUMPFUNUSDT"]
RSI_PERIODS = [5]
INTERVALS = ["30", "30"]

LONG_SWITCH_RSI = [28, 28]
SHORT_SWITCH_RSI = [72, 72]

LEVERAGE = "5"
PCT = 80
ENTRY_BAND = 4
COOLDOWN_BARS = 0

position = {s: None for s in SYMBOLS}
entry_px = {s: None for s in SYMBOLS}
last_peak_level = {s: None for s in SYMBOLS}
last_trough_level = {s: None for s in SYMBOLS}
armed_short_switch = {s: False for s in SYMBOLS}
armed_long_switch = {s: False for s in SYMBOLS}
last_closed_price1 = {s: None for s in SYMBOLS}
cooldown_bars = {s: 0 for s in SYMBOLS}

pending_floor_level = {s: None for s in SYMBOLS}
pending_ceiling_level = {s: None for s in SYMBOLS}

BASE_CASH = None

bybit.PCT = PCT
for s in SYMBOLS:
    if s not in bybit.SYMBOLS:
        bybit.SYMBOLS.append(s)

def start():
    global BASE_CASH
    BASE_CASH = bybit.get_usdt()
    print(f"🔧 보유($): {BASE_CASH:.2f} USDT")
    for s in SYMBOLS:
        bybit.set_leverage(symbol=s, leverage=LEVERAGE)

def update():
    while True:
        for idx, symbol in enumerate(SYMBOLS):
            try:
                rsi_period = RSI_PERIODS[idx]
                interval = INTERVALS[idx]
                long_rsi = LONG_SWITCH_RSI[idx]
                short_rsi = SHORT_SWITCH_RSI[idx]
                leverage = LEVERAGE

                pnl = bybit.get_PnL(symbol)
                roe = bybit.get_ROE(symbol)

                c_prev2, c_prev1, cur_3 = bybit.get_close_price(symbol, interval=interval)
                rsi = bybit.get_RSI(symbol, interval=interval, period=rsi_period)

                if rsi <= long_rsi:
                    armed_long_switch[symbol] = True
                if rsi >= short_rsi:
                    armed_short_switch[symbol] = True

                new_bar = (last_closed_price1[symbol] is None) or (last_closed_price1[symbol] != c_prev1)
                if new_bar:
                    last_closed_price1[symbol] = c_prev1
                    if cooldown_bars[symbol] > 0:
                        cooldown_bars[symbol] -= 1

                if rsi >= 84:
                    last_peak_level[symbol] = 84
                elif rsi >= 80:
                    if last_peak_level[symbol] is None or last_peak_level[symbol] < 80:
                        last_peak_level[symbol] = 80
                elif rsi >= 75:
                    if last_peak_level[symbol] is None or last_peak_level[symbol] < 75:
                        last_peak_level[symbol] = 75
                elif rsi >= 70:
                    if last_peak_level[symbol] is None or last_peak_level[symbol] < 70:
                        last_peak_level[symbol] = 70

                if rsi <= 20:
                    last_trough_level[symbol] = 20
                elif rsi <= 25:
                    if last_trough_level[symbol] is None or last_trough_level[symbol] > 25:
                        last_trough_level[symbol] = 25
                elif rsi <= 30:
                    if last_trough_level[symbol] is None or last_trough_level[symbol] > 30:
                        last_trough_level[symbol] = 30
                elif rsi <= 35:
                    if last_trough_level[symbol] is None or last_trough_level[symbol] > 35:
                        last_trough_level[symbol] = 35

                if position[symbol] is None and cooldown_bars[symbol] == 0:
                    if last_peak_level[symbol] is not None and armed_short_switch[symbol]:
                        short_trigger = last_peak_level[symbol] - 3
                        if (rsi <= short_trigger) and (rsi >= short_trigger - ENTRY_BAND):
                            px, qty = bybit.entry_position(symbol=symbol, side="Sell", leverage=leverage)
                            if qty > 0 and px is not None:
                                position[symbol] = "short"
                                entry_px[symbol] = px
                                cooldown_bars[symbol] = COOLDOWN_BARS
                                last_peak_level[symbol] = None
                                pending_floor_level[symbol] = None
                                armed_short_switch[symbol] = False
                                armed_long_switch[symbol] = (rsi <= long_rsi)
                                print(f"📉 {symbol} SHORT 진입 | entry={px}")
                                continue

                    if position[symbol] is None and last_trough_level[symbol] is not None and cooldown_bars[symbol] == 0 and armed_long_switch[symbol]:
                        long_trigger = last_trough_level[symbol] + 3
                        if (rsi >= long_trigger) and (rsi <= long_trigger + ENTRY_BAND):
                            px, qty = bybit.entry_position(symbol=symbol, side="Buy", leverage=leverage)
                            if qty > 0 and px is not None:
                                position[symbol] = "long"
                                entry_px[symbol] = px
                                cooldown_bars[symbol] = COOLDOWN_BARS
                                last_trough_level[symbol] = None
                                pending_ceiling_level[symbol] = None
                                armed_long_switch[symbol] = False
                                armed_short_switch[symbol] = (rsi >= short_rsi)
                                print(f"📈 {symbol} LONG 진입 | entry={px}")
                                continue

                elif position[symbol] == "short":
                    if rsi <= 30:
                        pending_floor_level[symbol] = 30 if pending_floor_level[symbol] is None else min(pending_floor_level[symbol], 30)
                    if rsi <= 25:
                        pending_floor_level[symbol] = 25 if pending_floor_level[symbol] is None else min(pending_floor_level[symbol], 25)
                    if rsi <= 20:
                        pending_floor_level[symbol] = 20 if pending_floor_level[symbol] is None else min(pending_floor_level[symbol], 20)
                    if rsi <= 15:
                        pending_floor_level[symbol] = 15 if pending_floor_level[symbol] is None else min(pending_floor_level[symbol], 15)

                    if pending_floor_level[symbol] is not None:
                        trigger_up = pending_floor_level[symbol] + 3
                        if rsi >= trigger_up:
                            bybit.close_position(symbol=symbol, side="Buy")
                            position[symbol] = None
                            entry_px[symbol] = None
                            cooldown_bars[symbol] = COOLDOWN_BARS

                            if rsi <= long_rsi:
                                px, qty = bybit.entry_position(symbol=symbol, side="Buy", leverage=leverage)
                                if qty > 0 and px is not None:
                                    position[symbol] = "long"
                                    entry_px[symbol] = px
                                    cooldown_bars[symbol] = COOLDOWN_BARS
                                    pending_floor_level[symbol] = None
                                    pending_ceiling_level[symbol] = None
                                    last_trough_level[symbol] = None
                                    armed_long_switch[symbol] = False
                                    armed_short_switch[symbol] = (rsi >= short_rsi)
                                    print(f"🔄 {symbol} SHORT 청산 후 LONG 전환 | entry={px}")
                                    continue

                            pending_floor_level[symbol] = None
                            pending_ceiling_level[symbol] = None
                            print(f"✅ {symbol} SHORT 청산")

                elif position[symbol] == "long":
                    if rsi >= 70:
                        pending_ceiling_level[symbol] = 70 if pending_ceiling_level[symbol] is None else max(pending_ceiling_level[symbol], 70)
                    if rsi >= 75:
                        pending_ceiling_level[symbol] = 75 if pending_ceiling_level[symbol] is None else max(pending_ceiling_level[symbol], 75)
                    if rsi >= 80:
                        pending_ceiling_level[symbol] = 80 if pending_ceiling_level[symbol] is None else max(pending_ceiling_level[symbol], 80)
                    if rsi >= 85:
                        pending_ceiling_level[symbol] = 85 if pending_ceiling_level[symbol] is None else max(pending_ceiling_level[symbol], 85)

                    if pending_ceiling_level[symbol] is not None:
                        trigger_down = pending_ceiling_level[symbol] - 3
                        if rsi <= trigger_down:
                            bybit.close_position(symbol=symbol, side="Sell")
                            position[symbol] = None
                            entry_px[symbol] = None
                            cooldown_bars[symbol] = COOLDOWN_BARS

                            if rsi >= short_rsi:
                                px, qty = bybit.entry_position(symbol=symbol, side="Sell", leverage=leverage)
                                if qty > 0 and px is not None:
                                    position[symbol] = "short"
                                    entry_px[symbol] = px
                                    cooldown_bars[symbol] = COOLDOWN_BARS
                                    pending_ceiling_level[symbol] = None
                                    pending_floor_level[symbol] = None
                                    last_peak_level[symbol] = None
                                    armed_short_switch[symbol] = False
                                    armed_long_switch[symbol] = (rsi <= long_rsi)
                                    print(f"🔄 {symbol} LONG 청산 후 SHORT 전환 | entry={px}")
                                    continue

                            pending_ceiling_level[symbol] = None
                            pending_floor_level[symbol] = None
                            print(f"✅ {symbol} LONG 청산")

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"🪙{symbol} @{interval} "
                    f"💲현재가: {cur_3:.5f}$ "
                    f"🚩포지션 {position.get(symbol)} "
                    f"| RSI({rsi_period})={rsi:.2f} "
                    f"| PnL: {pnl:.3f} "
                    f"| ROE: {roe:.2f}"
                )

            except Exception as e:
                print(f"[ERR] {symbol}: {type(e).__name__} {e}")
                continue

            time.sleep(5)
        time.sleep(10)

start()
update()