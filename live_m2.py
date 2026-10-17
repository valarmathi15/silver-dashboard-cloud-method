"""
live_m2.py  -  LIVE Method 2 (Cloud Scale-In, Google-sheet logic)
=================================================================
ONE single file - no m.py needed. The Method 2 engine is copied inside (same logic as m.py),
so the live numbers match the backtest dashboard.

What it does every refresh
  1. pulls candles from Zerodha Kite Connect (API key)       -> fetch_candles()
  2. drops the candle that is still forming (only CLOSED candles are used)
  3. runs M2 from your chosen start date (cloud is still computed on the warm-up history)
  4. shows: current lots / avg price / unrealised %, last signal (BUY / EXIT / HOLD),
     the bar-by-bar log and the trade list
  5. optional phone push through ntfy when a NEW BUY or EXIT appears on a closed candle

Run   :  python -m streamlit run live_m2.py
API key / secret : typed on the first screen (no Streamlit secrets needed)
"""
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

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


def run_m2(df, profit_target_pct, max_lots, start_date=None, lot_qty=1.0, trace=None):
    """Method 2 (Google-sheet logic). Returns trades DataFrame; trace (list) gets one dict per bar."""
    full = add_ichimoku_cloud(df)
    if start_date is not None:   # cloud uses full history, simulation starts flat from start_date
        full = full[full['Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    rows = full.to_dict('records')

    trades = []
    trade_id = 0
    m2_first, m2_lots, m2_avg, m2_prev_exit, m2_seg_start = True, 0, None, False, None

    for row in rows:
        if pd.isna(row['CloudTop']) or pd.isna(row['CloudBottom']):
            continue
        price, date = row['Close'], row['Date']
        above = bool(row['AboveCloud'])
        if m2_first:
            m2_first = False
            m2_lots = 1 if above else 0
            m2_avg = price if m2_lots == 1 else None
        else:
            prev_lots = m2_lots
            if not above:
                pass
            elif m2_prev_exit:
                m2_lots = 1
            elif m2_lots < max_lots:
                m2_lots += 1
            if m2_lots == 1:
                m2_avg = price
            elif m2_lots > prev_lots:
                m2_avg = (m2_avg * prev_lots + price) / m2_lots
        m2_prev_exit = False
        if m2_seg_start is None and m2_lots > 0:
            m2_seg_start = date
        if m2_lots > 0 and m2_avg:
            if (price - m2_avg) / m2_avg >= profit_target_pct / 100.0:
                trade_id += 1
                trades.append({
                    'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
                    'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
                    'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
                    'Exit Price': round(price, 4),
                    'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
                    'PnL (pts x qty)': round((price - m2_avg) * m2_lots * lot_qty, 2),
                    'Exit Reason': f'Profit target {profit_target_pct}% hit (Method 2 - sheet logic)'})
                m2_prev_exit = True
                m2_seg_start = None
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


# timeframe -> (Kite interval, raw bar length, warm-up calendar days needed for the cloud)
TF = {
    '1D': ('day',      pd.Timedelta(days=1),   260),
    '4H': ('60minute', pd.Timedelta(hours=1),  160),   # 4H = 4 hourly bars, built per day
    '1H': ('60minute', pd.Timedelta(hours=1),  100),
}
STATE_FILE = os.path.join(HERE, 'last_signal.json')


def secret(name, default=''):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# =============================================================== DATA FEED (only this part is broker specific)
def kite_client(api_key, access_token):
    from kiteconnect import KiteConnect
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k


@st.cache_data(ttl=86400, show_spinner=False)
def _instrument_token(_kite, api_key, exchange, symbol):
    for r in _kite.instruments(exchange):
        if r['tradingsymbol'].upper() == symbol.strip().upper():
            return r['instrument_token']
    raise ValueError(f'{symbol} not found on {exchange}')


def fetch_candles(kite, api_key, exchange, symbol, tf, start_date):
    """-> raw candles (Date, Open, High, Low, Close), tz-naive IST, oldest first."""
    interval, _, warm_days = TF[tf]
    token = _instrument_token(kite, api_key, exchange, symbol)
    frm = pd.Timestamp(start_date) - pd.Timedelta(days=warm_days)
    to = pd.Timestamp.now()
    step = pd.Timedelta(days=1900 if interval == 'day' else 190)   # Kite per-request limits
    rows, cur = [], frm
    while cur < to:
        end = min(cur + step, to)
        rows += kite.historical_data(token, cur.to_pydatetime(), end.to_pydatetime(), interval)
        cur = end + pd.Timedelta(seconds=1)
        time.sleep(0.4)                                            # Kite: max ~3 requests/sec
    if not rows:
        raise ValueError('no candles returned')
    df = pd.DataFrame(rows).rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High',
                                            'low': 'Low', 'close': 'Close'})
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


# =============================================================== M2 live state (engine reused as-is)
def compute_m2(df, profit_pct, max_lots, start_date):
    trace = []
    trades = run_m2(df, profit_pct, max_lots, start_date=start_date, trace=trace)
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


# =============================================================== login (Kite access token is valid for 1 day)
def get_kite(api_key, api_secret):
    """Login with the keys typed on screen. Token is kept only in THIS browser session."""
    token = st.session_state.get('kite_token', '')
    if token:
        try:
            k = kite_client(api_key, token)
            k.profile()
            return k
        except Exception:
            st.session_state['kite_token'] = ''
    from kiteconnect import KiteConnect
    kc = KiteConnect(api_key=api_key)
    st.markdown(f'1) [Login to Zerodha]({kc.login_url()})   2) copy `request_token` from the redirected URL   3) paste below')
    rt = st.text_input('request_token')
    if rt:
        try:
            sess = kc.generate_session(rt.strip(), api_secret=api_secret)
            st.session_state['kite_token'] = sess['access_token']
            st.rerun()
        except Exception as e:
            st.error(f'Login failed: {e}')
    return None


# =============================================================== UI
def main():
    st.set_page_config(page_title='Live Method 2', layout='wide')
    st.title('Live Method 2 - Cloud Scale-In')

    # ---- STEP 1: ask for the API key + secret on screen (nothing is stored on the server)
    if not st.session_state.get('keys_ok'):
        st.subheader('Enter your Kite API details')
        k_in = st.text_input('API key', type='password')
        s_in = st.text_input('API secret', type='password')
        if st.button('Continue') and k_in and s_in:
            st.session_state['api_key'], st.session_state['api_secret'] = k_in.strip(), s_in.strip()
            st.session_state['keys_ok'] = True
            st.rerun()
        return
    api_key, api_secret = st.session_state['api_key'], st.session_state['api_secret']

    with st.sidebar:
        if st.button('Change API key / logout'):
            for k_ in ('keys_ok', 'api_key', 'api_secret', 'kite_token'):
                st.session_state.pop(k_, None)
            st.rerun()
        exchange = st.selectbox('Exchange', ['NSE', 'MCX', 'NFO', 'BSE'])
        symbol = st.text_input('Trading symbol', 'NIFTY 50',
                               help='Kite tradingsymbol, e.g. NIFTY 50, RELIANCE, GOLDM26DECFUT')
        tf = st.selectbox('Timeframe', list(TF))
        profit = st.number_input('Profit target %', 0.5, 50.0, 1.0, 0.5)
        lots = st.number_input('Max lots', 1, 500, 10, 1)
        start = st.date_input('Start date (position starts flat here)', dt.date(dt.date.today().year, 1, 1))
        default_close = dt.time(23, 30) if exchange == 'MCX' else dt.time(15, 30)
        close_t = st.time_input('Market close', default_close)
        refresh = st.number_input('Refresh every (seconds)', 15, 3600, 60, 15)
        topic = st.text_input('ntfy topic (optional)')

    kite = get_kite(api_key, api_secret)
    if kite is None:
        return

    @st.cache_data(ttl=30, show_spinner=False)
    def load(_k, key_, ex, sym, tf_, start_):
        return fetch_candles(_k, api_key, ex, sym, tf_, start_)

    def render():
        try:
            raw = load(kite, api_key, exchange, symbol, tf, start)
        except Exception as e:
            st.error(f'Data error: {e}')
            return
        now = pd.Timestamp.now()
        df = prepare(raw, tf, close_t, now)
        tr, trades = compute_m2(df, profit, int(lots), start)
        if tr.empty:
            st.info('Not enough candles yet for the cloud (needs 78+ bars).')
            return
        last = tr.iloc[-1]
        flat = last['Signal'] in ('EXIT', 'BUY+EXIT')
        notify_new_signal(topic, f'{exchange}:{symbol}:{tf}', last)

        c = st.columns(6)
        c[0].metric('Last closed bar', str(last['Date'])[:16])
        c[1].metric('Close', f"{last['Close']:.2f}")
        c[2].metric('Cloud top / bottom', f"{last['CloudTop']:.2f} / {last['CloudBottom']:.2f}")
        c[3].metric('Position', 'Flat (exited)' if flat else f"{int(last['Lots'])} lots")
        c[4].metric('Avg entry', '-' if flat or pd.isna(last['AvgPrice']) else f"{last['AvgPrice']:.2f}")
        c[5].metric('Signal', last['Signal'],
                    None if flat or pd.isna(last['Unrealised %']) else f"{last['Unrealised %']:.2f}%")
        st.caption(f"Refreshed {now:%H:%M:%S}  |  only closed candles used  |  {len(df)} candles loaded")

        view = tr.tail(150).set_index('Date')
        st.line_chart(view[['Close', 'CloudTop', 'CloudBottom']])
        t1, t2 = st.tabs(['Bar log (latest first)', 'Trades'])
        with t1:
            st.dataframe(tr.sort_values('Date', ascending=False).head(100), use_container_width=True, hide_index=True)
        with t2:
            if trades.empty:
                st.write('No trades yet.')
            else:
                st.dataframe(trades.sort_values('Exit Date', ascending=False), use_container_width=True, hide_index=True)
                closed = trades[~trades['Exit Reason'].str.startswith('Open at end')]
                st.write(f"Closed trades: {len(closed)}   |   Closed P&L: {closed['PnL (pts x qty)'].sum():,.2f}"
                         f"   |   Open P&L: {trades['PnL (pts x qty)'].sum() - closed['PnL (pts x qty)'].sum():,.2f}")

    st.fragment(run_every=int(refresh))(render)()


if __name__ == '__main__':
    main()
