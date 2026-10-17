"""
live_m2.py  -  LIVE Method 2 (Cloud Scale-In, Google-sheet logic)
=================================================================
Same engine as the dashboard: it calls m.run_cloud_backtest(..., method='M2') from m.py
(m.py is NOT changed), so the live numbers always match the backtest dashboard.

What it does every refresh
  1. pulls candles from Zerodha Kite Connect (API key)       -> fetch_candles()
  2. drops the candle that is still forming (only CLOSED candles are used)
  3. runs M2 from your chosen start date (cloud is still computed on the warm-up history)
  4. shows: current lots / avg price / unrealised %, last signal (BUY / EXIT / HOLD),
     the bar-by-bar log and the trade list
  5. optional phone push through ntfy when a NEW BUY or EXIT appears on a closed candle

Run   :  python -m streamlit run live_m2.py        (keep m.py in the same folder)
Secrets (.streamlit/secrets.toml  or  environment variables)
    KITE_API_KEY    = "xxxx"
    KITE_API_SECRET = "xxxx"
    NTFY_TOPIC      = "your-topic"        # optional
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m as engine  # your engine file (unchanged)

# timeframe -> (Kite interval, raw bar length, warm-up calendar days needed for the cloud)
TF = {
    '1D': ('day',      pd.Timedelta(days=1),   260),
    '4H': ('60minute', pd.Timedelta(hours=1),  160),   # 4H = 4 hourly bars, built per day
    '1H': ('60minute', pd.Timedelta(hours=1),  100),
}
TOKEN_FILE = os.path.join(HERE, 'kite_token.json')
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
    trades = engine.run_cloud_backtest(df, profit_pct, max_lots, start_date=start_date,
                                       method='M2', trace=trace)
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
    token = secret('KITE_ACCESS_TOKEN') or st.session_state.get('kite_token', '')
    if not token and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            d = json.load(f)
        if d.get('date') == str(dt.date.today()):
            token = d['token']
    if token:
        try:
            k = kite_client(api_key, token)
            k.profile()
            return k
        except Exception:
            token = ''
    from kiteconnect import KiteConnect
    kc = KiteConnect(api_key=api_key)
    st.warning('Kite login needed (once per day).')
    st.markdown(f'1) [Login to Zerodha]({kc.login_url()})   2) copy `request_token` from the redirected URL   3) paste below')
    rt = st.text_input('request_token')
    if rt and api_secret:
        try:
            sess = kc.generate_session(rt.strip(), api_secret=api_secret)
            st.session_state['kite_token'] = sess['access_token']
            with open(TOKEN_FILE, 'w') as f:
                json.dump({'date': str(dt.date.today()), 'token': sess['access_token']}, f)
            st.rerun()
        except Exception as e:
            st.error(f'Login failed: {e}')
    return None


# =============================================================== UI
def main():
    st.set_page_config(page_title='Live Method 2', layout='wide')
    st.title('Live Method 2 - Cloud Scale-In')

    api_key, api_secret = secret('KITE_API_KEY'), secret('KITE_API_SECRET')
    if not api_key:
        st.error('KITE_API_KEY not set (secrets.toml / environment).')
        return

    with st.sidebar:
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
        topic = st.text_input('ntfy topic (optional)', secret('NTFY_TOPIC'))

    kite = get_kite(api_key, api_secret)
    if kite is None:
        return

    @st.cache_data(ttl=30, show_spinner=False)
    def load(_k, ex, sym, tf_, start_):
        return fetch_candles(_k, api_key, ex, sym, tf_, start_)

    def render():
        try:
            raw = load(kite, exchange, symbol, tf, start)
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
