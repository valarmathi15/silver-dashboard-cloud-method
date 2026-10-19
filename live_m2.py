"""
live_m2.py  -  LIVE PAPER TRADING : Method 2 (Cloud Scale-In, Google-sheet logic)   [single file]
=====================================================================================
BUILDER   : pick folder, exchange, symbols, candle timeframe, capital, qty, Max lots (many) and
            Target % (many)  ->  press Start  ->  ONE FILE is saved for every symbol x lots x target
            (e.g. folder "Nifty test":  Nifty 50 4H 10lots 1%.json,  ... 10lots 2%.json, ...)
MY STRATEGIES : open a folder -> summary table of all its files, all running live side by side;
            or open any file(s) to see that strategy's live position, chart, bar log and trades.

How "live paper" works: Method 2 is fully determined by the candles from the strategy START time,
so every time you open / refresh, the state is rebuilt from Angel One candles (closed candles only).
No background process is needed for the numbers. (ntfy push alerts need the app to be open.)

Run : python -m streamlit run live_m2.py
Login details (API key, Client ID, MPIN, TOTP secret) are typed on the first screen.
Strategy files are saved in the  strategies/<ClientID>/<folder>/  folder next to this file.
NOTE: on Streamlit Cloud that folder is wiped when the app reboots - run it on your own PC to keep files.
"""
import datetime as dt
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT_ROOT = os.path.join(HERE, 'strategies')
STATE_FILE = os.path.join(HERE, 'last_signal.json')

# timeframe -> (Angel interval, raw bar length, warm-up calendar days needed for the cloud)
TF = {
    '1D': ('ONE_DAY',  pd.Timedelta(days=1),   260),
    '4H': ('ONE_HOUR', pd.Timedelta(hours=1),  160),   # 4H = 4 hourly bars, built per day
    '1H': ('ONE_HOUR', pd.Timedelta(hours=1),  100),
}
LOTS_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TARGET_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SPAN_OPTIONS = {'SPAN A': 'A', 'SPAN B': 'B', 'SPAN A & B': 'A&B', 'SPAN A OR B': 'AorB'}
EXIT_OPTIONS = TARGET_OPTIONS + list(SPAN_OPTIONS)               # the 14 options of "Profit target / Span exit"
CANDLE_OPTIONS = {'Every candle': 'every', 'Daily': 'daily', 'Weekly': 'weekly'}


def exit_label(x):
    return f'{x}%' if isinstance(x, (int, float)) else x


def exit_text(s):
    return s.get('exit_label') or f"{s['profit']}%"


def candle_text(s):
    return {v: k for k, v in CANDLE_OPTIONS.items()}.get(s.get('candle', 'every'), 'Every candle')


# =============================================================== ENGINE (M2 only, copied from m.py - same logic)
TENKAN_P, KIJUN_P, SENKOU_B_P, DISPLACEMENT = 9, 26, 52, 26


def add_ichimoku_cloud(df):
    df = df.copy()
    high, low = df['High'], df['Low']
    tenkan = (high.rolling(TENKAN_P).max() + low.rolling(TENKAN_P).min()) / 2
    kijun = (high.rolling(KIJUN_P).max() + low.rolling(KIJUN_P).min()) / 2
    df['SenkouA'] = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    df['SenkouB'] = ((high.rolling(SENKOU_B_P).max() + low.rolling(SENKOU_B_P).min()) / 2).shift(DISPLACEMENT)
    df['CloudTop'] = df[['SenkouA', 'SenkouB']].max(axis=1, skipna=False)
    df['CloudBottom'] = df[['SenkouA', 'SenkouB']].min(axis=1, skipna=False)
    df['AboveCloud'] = df['Close'] > df['CloudTop']
    return df


SPAN_RULES = {                       # exit rule for the "Span" options (price = bar Close)
    'A':    lambda p, a, b: p < a,
    'B':    lambda p, a, b: p < b,
    'A&B':  lambda p, a, b: p < a and p < b,      # below BOTH spans  (= below the cloud bottom)
    'AorB': lambda p, a, b: p < a or p < b,       # below EITHER span (= below the cloud top)
}
SPAN_TEXT = {'A': 'Span A', 'B': 'Span B', 'A&B': 'Span A & B', 'AorB': 'Span A or B'}


def run_m2(df, profit_target_pct, max_lots, start_date=None, lot_qty=1.0, trace=None, span=None, candle='every'):
    """Method 2 (Google-sheet logic).
    span   : None = exit on Profit target %;  'A' / 'B' / 'A&B' / 'AorB' = exit when Close falls below that span
             (whole position closed; span exits go flat, no repeat exits).
    candle : 'every' = buy 1 lot on every candle above the cloud;  'daily' = at most 1 lot per day;
             'weekly' = at most 1 lot per week (first candle of the period that closes above the cloud).
    Returns trades DataFrame; trace (list) gets one dict per bar."""
    full = add_ichimoku_cloud(df)
    if start_date is not None:   # cloud uses full history, simulation starts flat from start_date
        full = full[full['Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    rows = full.to_dict('records')
    span_hit = SPAN_RULES.get(span)
    original = candle == 'every' and span_hit is None      # exactly the sheet logic of m.py

    def pkey(d):
        if candle == 'daily':
            return d.date()
        if candle == 'weekly':
            y, w, _ = d.isocalendar()
            return (y, w)
        return None

    exit_reason = (f'Closed below {SPAN_TEXT[span]} (Method 2)' if span_hit
                   else f'Profit target {profit_target_pct}% hit (Method 2 - sheet logic)')
    trades = []
    trade_id = 0
    m2_first, m2_lots, m2_avg, m2_prev_exit, m2_seg_start = True, 0, None, False, None
    last_buy_key = None

    for row in rows:
        if pd.isna(row['CloudTop']) or pd.isna(row['CloudBottom']):
            continue
        price, date = row['Close'], row['Date']
        above = bool(row['AboveCloud'])
        if m2_first:
            m2_first = False
            m2_lots = 1 if above else 0
            m2_avg = price if m2_lots == 1 else None
            if m2_lots == 1:
                last_buy_key = pkey(date)
        else:
            prev_lots, bought = m2_lots, False
            can_buy = candle == 'every' or pkey(date) != last_buy_key
            if not above or not can_buy:
                pass
            elif m2_prev_exit:
                m2_lots, bought = 1, True
            elif m2_lots < max_lots:
                m2_lots, bought = m2_lots + 1, True
            if bought:
                last_buy_key = pkey(date)
            if original:                                   # sheet quirk: avg follows Close while lots == 1
                if m2_lots == 1:
                    m2_avg = price
                elif m2_lots > prev_lots:
                    m2_avg = (m2_avg * prev_lots + price) / m2_lots
            elif bought:
                if m2_lots == 1:
                    m2_avg = price
                else:
                    m2_avg = (m2_avg * prev_lots + price) / m2_lots
        m2_prev_exit = False
        if m2_seg_start is None and m2_lots > 0:
            m2_seg_start = date
        if m2_lots > 0 and m2_avg:
            if span_hit:
                hit = span_hit(price, row['SenkouA'], row['SenkouB'])
            else:
                hit = (price - m2_avg) / m2_avg >= profit_target_pct / 100.0
            if hit:
                trade_id += 1
                trades.append({
                    'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
                    'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
                    'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
                    'Exit Price': round(price, 4),
                    'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
                    'PnL (pts x qty)': round((price - m2_avg) * m2_lots * lot_qty, 2),
                    'Exit Reason': exit_reason})
                m2_prev_exit = True
                m2_seg_start = None
                if span_hit:                               # span exit = flat (no repeat exit on the old lots)
                    m2_lots, m2_avg = 0, None
        if trace is not None:
            trace.append({'Date': date, 'Close': price, 'CloudTop': row['CloudTop'],
                          'CloudBottom': row['CloudBottom'], 'Above': above, 'Lots': m2_lots,
                          'AvgPrice': None if not m2_avg else round(m2_avg, 4),
                          'Action': 'EXIT' if m2_prev_exit else ''})

    if m2_lots > 0 and m2_avg and not m2_prev_exit and m2_seg_start is not None and rows:
        price, date = full.iloc[-1]['Close'], full.iloc[-1]['Date']
        trade_id += 1
        trades.append({
            'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
            'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
            'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
            'Exit Price': round(price, 4),
            'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
            'PnL (pts x qty)': round((price - m2_avg) * m2_lots * lot_qty, 2),
            'Exit Reason': 'Open at end of data (marked to last close)'})
    return pd.DataFrame(trades)


# =============================================================== DATA FEED (Angel One SmartAPI)
SCRIP_URL = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'


def angel_login(client_id, api_key, mpin, totp_secret):
    """Login with Client ID + MPIN + TOTP (the 6-digit code is generated from your TOTP secret)."""
    import pyotp
    from SmartApi import SmartConnect
    smart = SmartConnect(api_key=api_key)
    res = smart.generateSession(client_id, mpin, pyotp.TOTP(totp_secret.replace(' ', '')).now())
    if not res or not res.get('status'):
        raise ValueError((res or {}).get('message', 'login failed'))
    return smart


@st.cache_data(ttl=86400, show_spinner='Loading Angel One instrument list ...')
def _scrip_master():
    return requests.get(SCRIP_URL, timeout=120).json()


def _instrument_token(exchange, symbol):
    sym = symbol.strip().upper()
    hits = [r for r in _scrip_master() if r['exch_seg'] == exchange
            and r['symbol'].upper() in (sym, sym + '-EQ')]
    if not hits:
        raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')
    hits.sort(key=lambda r: r['symbol'].upper() != sym)          # exact match first
    return hits[0]['token']


def fetch_candles(smart, exchange, symbol, tf, start_date):
    """-> raw candles (Date, Open, High, Low, Close), tz-naive IST, oldest first."""
    interval, _, warm_days = TF[tf]
    token = _instrument_token(exchange, symbol)
    frm = pd.Timestamp(start_date) - pd.Timedelta(days=warm_days)
    to = pd.Timestamp.now()
    step = pd.Timedelta(days=1900 if interval == 'ONE_DAY' else 190)   # Angel per-request limits
    rows, cur = [], frm
    while cur < to:
        end = min(cur + step, to)
        res = smart.getCandleData({'exchange': exchange, 'symboltoken': token, 'interval': interval,
                                   'fromdate': cur.strftime('%Y-%m-%d %H:%M'),
                                   'todate': end.strftime('%Y-%m-%d %H:%M')})
        if not res or not res.get('status'):
            raise ValueError((res or {}).get('message', 'candle request failed'))
        rows += res.get('data') or []
        cur = end + pd.Timedelta(minutes=1)
        time.sleep(0.5)                                                # Angel: ~3 requests/sec
    if not rows:
        raise ValueError('no candles returned')
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']].drop_duplicates('Date', keep='last')
    return df.sort_values('Date').reset_index(drop=True)


# =============================================================== candle preparation
def closed_only(df, delta, close_t, now):
    """drop the candle that has not finished yet (its end = start+length, capped at market close)."""
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + delta, sess_end], axis=1).min(axis=1)
    return df[end <= now].reset_index(drop=True)


def to_4h(h, close_t):
    """hourly -> 4H: per day, bars 1-4 = first 4H candle, the rest = second (e.g. NSE 09:15 + 13:15)."""
    h = h.copy()
    h['day'] = h['Date'].dt.normalize()
    h['n'] = h.groupby('day').cumcount() // 4
    out = (h.groupby(['day', 'n'])
             .agg(Date=('Date', 'first'), Open=('Open', 'first'), High=('High', 'max'),
                  Low=('Low', 'min'), Close=('Close', 'last'), cnt=('Close', 'size'), last=('Date', 'last'))
             .reset_index(drop=True))
    day_close = out['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    ok = (out['cnt'] == 4) | ((out['last'] + pd.Timedelta(hours=1)) >= day_close)
    return out[ok][['Date', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)


def prepare(raw, tf, close_t, now):
    _, delta, _ = TF[tf]
    df = closed_only(raw, delta, close_t, now)
    return to_4h(df, close_t) if tf == '4H' else df


# =============================================================== M2 live state
def compute_m2(df, profit_pct, max_lots, start, qty=1.0, span=None, candle='every'):
    trace = []
    trades = run_m2(df, profit_pct, max_lots, start_date=start, lot_qty=qty, trace=trace, span=span, candle=candle)
    tr = pd.DataFrame(trace)
    if tr.empty:
        return tr, trades
    prev_lots = tr['Lots'].shift(fill_value=0)
    prev_exit = tr['Action'].shift(fill_value='') == 'EXIT'
    buy = tr['Above'] & ((tr['Lots'] > prev_lots) | prev_exit)
    exit_ = tr['Action'] == 'EXIT'
    tr['Signal'] = np.select([buy & exit_, exit_, buy], ['BUY+EXIT', 'EXIT', 'BUY'], 'HOLD')
    tr['Unrealised %'] = np.where(tr['AvgPrice'].notna(),
                                  (tr['Close'] - tr['AvgPrice']) / tr['AvgPrice'] * 100, np.nan).round(2)
    return tr, trades


# =============================================================== ntfy (only on a NEW signal)
def notify_new_signal(topic, key, last_row):
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
    bar = str(last_row['Date'])
    first_time = key not in state
    if state.get(key) != bar:
        state[key] = bar
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        if not first_time and topic and last_row['Signal'] != 'HOLD':     # first run = baseline only
            msg = (f"{key}: {last_row['Signal']} @ {last_row['Close']:.2f}  |  lots {int(last_row['Lots'])}"
                   f"  avg {last_row['AvgPrice']}  |  bar {bar}")
            try:
                requests.post(f'https://ntfy.sh/{topic}', data=msg.encode('utf-8'),
                              headers={'Title': 'Method 2 signal'}, timeout=10)
            except Exception as e:
                st.warning(f'ntfy failed: {e}')


def get_smart():
    """Logged-in SmartConnect for THIS browser session (re-logs in automatically if it expired)."""
    c = st.session_state['creds']
    if 'smart' not in st.session_state:
        st.session_state['smart'] = angel_login(c['client_id'], c['api_key'], c['mpin'], c['totp'])
    return st.session_state['smart']


def relogin():
    st.session_state.pop('smart', None)
    return get_smart()


# =============================================================== strategy files (folder / file)
def safe(s):
    return re.sub(r'[^A-Za-z0-9 _.%\-]', '_', str(s)).strip() or 'x'


def user_dir():
    return os.path.join(STRAT_ROOT, safe(st.session_state['creds']['client_id']))


def list_folders():
    d = user_dir()
    return sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))) if os.path.isdir(d) else []


def list_files(folder):
    d = os.path.join(user_dir(), safe(folder))
    return sorted(f for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else []


def load_strategy(folder, fname):
    with open(os.path.join(user_dir(), safe(folder), fname), encoding='utf-8') as f:
        return json.load(f)


def delete_strategy(folder, fname):
    os.remove(os.path.join(user_dir(), safe(folder), fname))


def create_strategies(folder, prefix, exchange, symbols, tf, capital, qty, close, lots_list, exits, candles, start):
    """one file per symbol x lots x exit option x candle mode. Existing file is NOT overwritten (keeps its start time)."""
    d = os.path.join(user_dir(), safe(folder))
    os.makedirs(d, exist_ok=True)
    made, skipped = [], []
    for sym in symbols:
        for lots in lots_list:
            for ex in exits:
                for cd in candles:
                    cdc = CANDLE_OPTIONS[cd]
                    name = (f"{prefix + ' ' if prefix else ''}{sym} {tf} {lots}lots {exit_label(ex)}"
                            + ('' if cdc == 'every' else f' {cdc}'))
                    path = os.path.join(d, safe(name) + '.json')
                    if os.path.exists(path):
                        skipped.append(name)
                        continue
                    is_span = ex in SPAN_OPTIONS
                    s = {'name': name, 'folder': folder, 'mode': 'paper', 'exchange': exchange, 'symbol': sym,
                         'tf': tf, 'profit': None if is_span else ex, 'span': SPAN_OPTIONS.get(ex),
                         'exit_label': exit_label(ex), 'candle': cdc, 'lots': lots, 'capital': capital,
                         'qty': qty, 'close': close, 'start': start,
                         'created': dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(s, f, indent=1)
                    made.append(name)
    return made, skipped


# =============================================================== evaluate one saved strategy
@st.cache_data(ttl=30, show_spinner=False)
def _load_candles(_smart, cid, exchange, symbol, tf, start_day):
    return fetch_candles(_smart, exchange, symbol, tf, start_day)   # same symbol/tf/day => one fetch for all files


def get_candles(s):
    cid = st.session_state['creds']['client_id']
    day = pd.Timestamp(s['start']).normalize().strftime('%Y-%m-%d')
    try:
        return _load_candles(get_smart(), cid, s['exchange'], s['symbol'], s['tf'], day)
    except Exception:
        return _load_candles(relogin(), cid, s['exchange'], s['symbol'], s['tf'], day)   # session expired?


def evaluate(s):
    """-> dict with tr (bar log), trades and the numbers shown in the summary."""
    h, m = [int(x) for x in s['close'].split(':')]
    raw = get_candles(s)
    df = prepare(raw, s['tf'], dt.time(h, m), pd.Timestamp.now())
    tr, trades = compute_m2(df, s.get('profit') or 0.0, int(s['lots']), s['start'], s['qty'],
                            span=s.get('span'), candle=s.get('candle', 'every'))
    out = {'s': s, 'tr': tr, 'trades': trades, 'status': 'Waiting for 1st candle', 'last': None,
           'closed_n': 0, 'closed_pnl': 0.0, 'open_pnl': 0.0, 'invested': 0.0}
    if tr.empty:
        return out
    last = tr.iloc[-1]
    flat = last['Signal'] in ('EXIT', 'BUY+EXIT') or int(last['Lots']) == 0 or pd.isna(last['AvgPrice'])
    openrows = trades[trades['Exit Reason'].str.startswith('Open at end')] if not trades.empty else trades
    closed = trades.drop(openrows.index) if not trades.empty else trades
    out.update(last=last, flat=flat, closed_n=len(closed),
               closed_pnl=float(closed['PnL (pts x qty)'].sum()) if len(closed) else 0.0,
               open_pnl=0.0 if flat or openrows.empty else float(openrows['PnL (pts x qty)'].sum()),
               invested=0.0 if flat else float(last['Lots']) * float(last['AvgPrice']) * float(s['qty']),
               status='Flat' if flat else f"Holding {int(last['Lots'])} lots")
    return out


def money(x):
    return f"{x:,.0f}"


# =============================================================== pages
def checkbox_dropdown(title, options, default, key, fmt=str):
    """Drop-down with a check box beside every option + Select all. Returns the ticked options."""
    base = f'cb_{key}'
    for o in options:
        if f'{base}_{o}' not in st.session_state:
            st.session_state[f'{base}_{o}'] = o in default

    def _toggle_all():
        v = st.session_state[f'{base}__all']
        for o in options:
            st.session_state[f'{base}_{o}'] = v

    chosen = [o for o in options if st.session_state[f'{base}_{o}']]
    st.session_state[f'{base}__all'] = len(chosen) == len(options)
    with st.popover(f'{title} · {len(chosen)} selected', width='stretch'):
        st.checkbox('Select all', key=f'{base}__all', on_change=_toggle_all)
        for o in options:
            st.checkbox(fmt(o), key=f'{base}_{o}')
    return chosen


NIFTY50_STOCKS = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                  'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL',
                  'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDIGO',
                  'INFY', 'ITC', 'JIOFIN', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
                  'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TATACONSUMER',
                  'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']


@st.cache_data(ttl=86400, show_spinner='Loading instrument list ...')
def instrument_choices(exchange):
    """{'Index': [...], 'Stocks': [...], 'Commodity': [...]} for the drop-downs (built from Angel's instrument list)."""
    try:
        master = _scrip_master()
    except Exception:
        return {}
    rows = [r for r in master if r.get('exch_seg') == exchange]
    out = {}
    idx = sorted({r['symbol'] for r in rows if r.get('instrumenttype') == 'AMXIDX'})
    if idx:
        out['Index'] = idx
    if exchange == 'NSE':
        eq = {r['symbol'] for r in rows if r['symbol'].endswith('-EQ')}
        st_ = [x for x in NIFTY50_STOCKS if x + '-EQ' in eq]
        if st_:
            out['Stocks'] = st_
    if exchange == 'MCX':
        today, best = dt.datetime.now(), {}
        for r in rows:
            if r.get('instrumenttype') != 'FUTCOM':
                continue
            try:
                exp = dt.datetime.strptime(r['expiry'], '%d%b%Y')
            except Exception:
                continue
            if exp.date() >= today.date() and (r['name'] not in best or exp < best[r['name']][0]):
                best[r['name']] = (exp, r['symbol'])
        if best:
            out['Commodity'] = [v[1] for k, v in sorted(best.items())]      # nearest-expiry future of each commodity
    return out


def page_builder():
    st.subheader('Strategy builder - Method 2')
    st.caption('Paper trading. Every Symbol x Max lots x Target % you pick is saved as its own file inside the folder.')
    folders = list_folders()
    c1, c2 = st.columns(2)
    with c1:
        pick = st.selectbox('Folder', ['+ New folder'] + folders)
        folder = st.text_input('New folder name', placeholder='e.g. Nifty 4H test') if pick == '+ New folder' else pick
    with c2:
        prefix = st.text_input('File name prefix (optional)', placeholder='e.g. morning')
    c1, c2, c3 = st.columns(3)
    exchange = c1.selectbox('Exchange', ['NSE', 'MCX', 'NFO', 'BSE'])
    capital = c2.number_input('Capital', min_value=0.0, value=1000000.0, step=10000.0)
    tf = c3.selectbox('Candle timeframe', list(TF))

    st.markdown('**Entry condition**  - Ichimoku cloud, Method 2')
    picked = []
    for label, opts in instrument_choices(exchange).items():
        st.caption(f'{label} ({len(opts)})')
        picked += checkbox_dropdown(f'Select {label.lower()}', opts, [], f'sym_{exchange}_{label}')
    typed_txt = st.text_input('Or type names (comma separated)', placeholder='e.g. Nifty 50, RELIANCE, GOLDM05DEC25FUT',
                              help='Index: Nifty 50  |  Stock: RELIANCE (or RELIANCE-EQ)  |  MCX: exact future symbol. '
                                   'Typed names are added to the ticked ones.')
    c1, c2 = st.columns(2)
    with c1:
        st.caption('Max lots (one file each)')
        lots_list = checkbox_dropdown('Select Max lots', LOTS_OPTIONS, [10], 'lots')
    qty = c2.number_input('Quantity per lot', min_value=1.0, value=1.0, step=1.0)
    st.markdown('**Candle**')
    st.caption('Buy on  (one file each)')
    candles = checkbox_dropdown('Select candle', list(CANDLE_OPTIONS), ['Every candle'], 'candle')
    st.markdown('**Exit condition**')
    st.caption(f'Profit target / Span exit ({len(EXIT_OPTIONS)})  (one file each)')
    targets = checkbox_dropdown('Select profit target / span exit', EXIT_OPTIONS, [1, 2], 'exits', fmt=exit_label)
    c1, c2, c3 = st.columns(3)
    close_t = c1.time_input('Market close', dt.time(23, 30) if exchange == 'MCX' else dt.time(15, 30), key=f'close_{exchange}')
    start_mode = c2.radio('Start from', ['Now', 'Pick date'], horizontal=True)
    if start_mode == 'Pick date':
        d0 = c3.date_input('Start date', dt.date.today() - dt.timedelta(days=30))
        start = pd.Timestamp(d0).strftime('%Y-%m-%d %H:%M')
    else:
        start = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')

    symbols, seen = [], set()
    for x in picked + [t.strip() for t in typed_txt.split(',') if t.strip()]:
        if x.upper() not in seen:
            seen.add(x.upper())
            symbols.append(x)
    n = len(symbols) * len(lots_list) * len(targets) * len(candles)
    st.caption(f'{len(symbols)} symbol(s) x {len(lots_list)} lots x {len(targets)} exits x {len(candles)} candle mode(s) = {n} file(s) will be saved.  Start = {start}  ({"first candle that closes after this" if start_mode == "Now" else "backfilled from that date"})')
    st.caption('Live trading (real orders) is not included - run Paper first and check the numbers.')
    if st.button('Start', type='primary'):
        if not folder or not folder.strip():
            st.error('Give a folder name.')
        elif n == 0:
            st.error('Pick (or type) at least one symbol, one Max lots, one candle option and one exit option.')
        elif n > 150:
            st.error('Too many files (max 150 at a time).')
        else:
            try:
                for sym in symbols:
                    _instrument_token(exchange, sym)          # fail early if a symbol is wrong
            except Exception as e:
                st.error(str(e))
                return
            made, skipped = create_strategies(folder.strip(), prefix.strip(), exchange, symbols, tf, capital, qty,
                                              close_t.strftime('%H:%M'), lots_list, targets, candles, start)
            if made:
                st.success(f'Saved {len(made)} file(s) in folder "{folder.strip()}". Open "My strategies" to watch them live.')
            if skipped:
                st.warning(f'{len(skipped)} file(s) already existed and were kept as they are (their start time is unchanged).')


def detail(ev):
    s, tr, trades, last = ev['s'], ev['tr'], ev['trades'], ev['last']
    st.markdown(f"#### {s['name']}")
    st.caption(f"{s['exchange']}:{s['symbol']}  |  {s['tf']}  |  {s['lots']} lots  |  exit {exit_text(s)}  |  buy {candle_text(s).lower()}  |  qty/lot {s['qty']:g}"
               f"  |  capital {money(s['capital'])}  |  started {s['start']}")
    if last is None:
        st.info('Waiting for the first closed candle after the start time (cloud needs 78+ bars of history).')
        return
    flat = ev['flat']
    c = st.columns(3)
    c[0].metric('Last closed bar', str(last['Date'])[:16])
    c[1].metric('Close', f"{last['Close']:.2f}")
    c[2].metric('Signal', last['Signal'])
    c = st.columns(3)
    c[0].metric('Position', ev['status'])
    c[1].metric('Avg entry', '-' if flat else f"{last['AvgPrice']:.2f}",
                None if flat or pd.isna(last['Unrealised %']) else f"{last['Unrealised %']:.2f}%")
    c[2].metric('Cloud top / bottom', f"{last['CloudTop']:.2f} / {last['CloudBottom']:.2f}")
    total = ev['closed_pnl'] + ev['open_pnl']
    c = st.columns(3)
    c[0].metric('Closed P&L', money(ev['closed_pnl']), f"{ev['closed_n']} trades", delta_color='off')
    c[1].metric('Open P&L', money(ev['open_pnl']))
    c[2].metric('Total P&L', money(total), f"{total / s['capital'] * 100:.2f}% of capital" if s['capital'] else None)
    st.line_chart(tr.tail(150).set_index('Date')[['Close', 'CloudTop', 'CloudBottom']])
    t1, t2 = st.tabs(['Bar log (latest first)', 'Trades'])
    with t1:
        st.dataframe(tr.sort_values('Date', ascending=False).head(100), width='stretch', hide_index=True)
    with t2:
        if trades.empty:
            st.write('No trades yet.')
        else:
            st.dataframe(trades.sort_values('Exit Date', ascending=False), width='stretch', hide_index=True)


def page_view(topic, refresh):
    def body():
        folders = list_folders()
        if not folders:
            st.info('No strategies yet. Create some in the Builder.')
            return
        folder = st.selectbox('Folder', folders)
        files = list_files(folder)
        if not files:
            st.info('This folder is empty.')
            return
        view = st.radio('View', ['Folder summary', 'Open file(s)'], horizontal=True)
        strategies = [load_strategy(folder, f) for f in files]
        evs = {}
        for f, s in zip(files, strategies):
            try:
                evs[f] = evaluate(s)
            except Exception as e:
                st.error(f"{s['name']}: {e}")
        for f, ev in evs.items():                                  # push alert on a NEW closed-candle signal
            if ev['last'] is not None:
                notify_new_signal(topic, f"{st.session_state['creds']['client_id']}/{folder}/{f}", ev['last'])
        st.caption(f"Refreshed {dt.datetime.now():%H:%M:%S}  |  closed candles only  |  every {refresh}s")

        if view == 'Folder summary':
            rows = []
            for f, ev in evs.items():
                s, last = ev['s'], ev['last']
                rows.append({'File': s['name'], 'Exit': exit_text(s), 'Candle': candle_text(s), 'Status': ev['status'],
                             'Avg entry': None if last is None or ev['flat'] else float(last['AvgPrice']),
                             'Close': None if last is None else float(last['Close']),
                             'Unrealised %': None if last is None or ev['flat'] else float(last['Unrealised %']),
                             'Open P&L': ev['open_pnl'], 'Closed trades': ev['closed_n'],
                             'Closed P&L': ev['closed_pnl'], 'Total P&L': ev['closed_pnl'] + ev['open_pnl'],
                             'Last signal': '-' if last is None else last['Signal'],
                             'Last bar': '-' if last is None else str(last['Date'])[:16]})
            if rows:
                tot = sum(r['Total P&L'] for r in rows)
                st.metric('Folder total P&L (all files)', money(tot))
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            return

        st.caption('Open file(s)')
        chosen = checkbox_dropdown('Select files', files, files[:1], f'files_{folder}', fmt=lambda f: f[:-5])
        layout = st.radio('Layout', ['Side by side', 'One below another'], horizontal=True) if len(chosen) > 1 else 'One below another'
        cols = st.columns(len(chosen)) if layout == 'Side by side' and len(chosen) > 1 else [st.container() for _ in chosen]
        for col, f in zip(cols, chosen):
            with col:
                if f in evs:
                    detail(evs[f])
                    if st.checkbox('Delete this file', key=f'del_{folder}_{f}'):
                        if st.button('Confirm delete', key=f'delbtn_{folder}_{f}'):
                            delete_strategy(folder, f)
                            st.rerun()

    st.fragment(run_every=int(refresh))(body)()


# =============================================================== app
def main():
    st.set_page_config(page_title='Live Paper Trading', layout='wide')
    st.title('Live Paper Trading - Method 2')

    # ---- login screen (nothing is stored on the server)
    if 'creds' not in st.session_state:
        st.subheader('Enter your Angel One details')
        api_key = st.text_input('API key', type='password')
        client_id = st.text_input('Client ID')
        mpin = st.text_input('MPIN', type='password')
        totp = st.text_input('TOTP secret key', type='password',
                             help='The long text shown when you enable TOTP (not the 6-digit code)')
        if st.button('Login') and api_key and client_id and mpin and totp:
            creds = {'api_key': api_key.strip(), 'client_id': client_id.strip().upper(),
                     'mpin': mpin.strip(), 'totp': totp.strip()}
            try:
                st.session_state['creds'] = creds
                get_smart()
                st.rerun()
            except Exception as e:
                st.session_state.pop('creds', None)
                st.session_state.pop('smart', None)
                st.error(f'Login failed: {e}')
        return

    with st.sidebar:
        with st.expander('Account', expanded=False):
            st.write(f"Client ID: {st.session_state['creds']['client_id']}")
            topic = st.text_input('ntfy topic (optional)')
            refresh = st.number_input('Refresh every (seconds)', 15, 3600, 60, 15)
            if st.button('Logout / change details'):
                for k_ in ('creds', 'smart'):
                    st.session_state.pop(k_, None)
                st.rerun()

    page = st.radio('Page', ['Builder', 'My strategies'], horizontal=True, label_visibility='collapsed')
    if page == 'Builder':
        page_builder()
    else:
        page_view(topic, refresh)


if __name__ == '__main__':
    main()
