

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.formula.api as smf

st.set_page_config(layout='wide', page_title='Trader Performance vs Sentiment', initial_sidebar_state='expanded')

# ---------------------- Helpers ----------------------
@st.cache_data
def prepare_data(trades_df, fg_df):
    # normalize col names
    trades = trades_df.copy()
    fg = fg_df.copy()

    # detect time column
    time_col = None
    for c in trades.columns:
        if c.lower() in ['time','timestamp','datetime','time_ms','exec_time']:
            time_col = c
            break
    if time_col is None:
        # fallback to first col that looks like date/time
        for c in trades.columns:
            if 'time' in c.lower() or 'date' in c.lower():
                time_col = c
                break
    if time_col is None:
        raise RuntimeError('No time column found in trades file.')

    # map a few common fields
    def find_col(df, keywords):
        for k in keywords:
            for c in df.columns:
                if k in c.lower():
                    return c
        return None

    pnl_col = find_col(trades, ['closedpnl','pnl','realized_pnl','closed_pnl'])
    acct_col = find_col(trades, ['account','acct','trader','user'])
    lev_col = find_col(trades, ['leverage','lev'])
    size_col = find_col(trades, ['size','qty','quantity'])
    side_col = find_col(trades, ['side','direction'])
    sym_col = find_col(trades, ['symbol','instrument','pair'])

    if not pnl_col or not acct_col:
        raise RuntimeError('Could not find essential columns (pnl/account) in trades file. Inspect column names.')

    trades = trades.rename(columns={time_col:'time', pnl_col:'closedPnL', acct_col:'account'})
    if lev_col: trades = trades.rename(columns={lev_col:'leverage'})
    if size_col: trades = trades.rename(columns={size_col:'size'})
    if side_col: trades = trades.rename(columns={side_col:'side'})
    if sym_col: trades = trades.rename(columns={sym_col:'symbol'})

    # parse time
    if np.issubdtype(trades['time'].dtype, np.number):
        sample = trades['time'].dropna().iloc[0]
        unit = 'ms' if sample > 1e12 else 's'
        trades['time_dt'] = pd.to_datetime(trades['time'], unit=unit, errors='coerce')
    else:
        trades['time_dt'] = pd.to_datetime(trades['time'], errors='coerce')
    trades['date'] = trades['time_dt'].dt.date

    # numeric conversions
    trades['closedPnL'] = pd.to_numeric(trades['closedPnL'], errors='coerce')
    trades['win'] = trades['closedPnL'] > 0
    if 'leverage' in trades.columns:
        trades['leverage'] = pd.to_numeric(trades['leverage'], errors='coerce')
    if 'size' in trades.columns:
        trades['size'] = pd.to_numeric(trades['size'], errors='coerce')

    # Fear & Greed
    date_col = None
    for c in fg.columns:
        if 'date' in c.lower():
            date_col = c
            break
    sent_col = None
    for c in fg.columns:
        if 'class' in c.lower() or 'sent' in c.lower() or 'fear' in c.lower() or 'greed' in c.lower():
            sent_col = c
            break
    if date_col is None:
        raise RuntimeError('No date column in fear/greed file')
    if sent_col is None:
        sent_col = fg.columns[1]

    fg = fg.rename(columns={date_col:'Date', sent_col:'sentiment'})
    fg['Date'] = pd.to_datetime(fg['Date'], errors='coerce').dt.date

    # daily aggregation
    agg_funcs = {'closedPnL': ['sum','mean','median'], 'account': pd.Series.nunique}
    if 'size' in trades.columns:
        agg_funcs['size'] = ['mean','median']
    if 'leverage' in trades.columns:
        agg_funcs['leverage'] = ['mean','median']

    daily = trades.groupby('date').agg(agg_funcs)
    daily.columns = ['_'.join(map(str,c)).strip() for c in daily.columns.values]
    daily['win_rate'] = trades.groupby('date')['win'].mean()
    daily = daily.reset_index().rename(columns={'date':'Date'})
    daily['Date'] = pd.to_datetime(daily['Date']).dt.date
    daily = daily.merge(fg[['Date','sentiment']], on='Date', how='left')

    # lagged sentiment columns
    daily_dt = pd.to_datetime(daily['Date'])
    mapping = dict(zip(fg['Date'], fg['sentiment']))
    for lag in [1,2,3,7]:
        lag_dates = (daily_dt - pd.Timedelta(days=lag)).dt.date
        daily[f'sentiment_lag_{lag}'] = [mapping.get(d, np.nan) for d in lag_dates]

    return trades, fg, daily

import os

# ---------------------- Data Loading ----------------------
st.sidebar.header('Data Status')

# Use direct file paths for local files
historical_file_path = 'https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing'
fg_file_path = 'fear_greed.csv'

if not os.path.exists(historical_file_path):
    st.error(f'Historical trades file not found: {historical_file_path}')
    st.stop()
if not os.path.exists(fg_file_path):
    st.error(f'Fear & Greed file not found: {fg_file_path}')
    st.stop()
    
st.sidebar.success(f'✅ Found: {historical_file_path}')
st.sidebar.success(f'✅ Found: {fg_file_path}')

# Load data
try:
    trades_df = pd.read_csv(historical_file_path)
    fg_df = pd.read_csv(fg_file_path)
    st.sidebar.success(f'✅ Loaded {len(trades_df)} trades and {len(fg_df)} sentiment records')
except Exception as e:
    st.error(f'Failed to load files: {e}')
    st.stop()

# Prepare
with st.spinner('Preparing data...'):
    trades, fg, daily = prepare_data(trades_df, fg_df)

# ---------------------- Controls ----------------------
st.sidebar.header('Filters')
min_date = pd.to_datetime(daily['Date']).min().date()
max_date = pd.to_datetime(daily['Date']).max().date()

date_range = st.sidebar.date_input('Date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)

accounts = sorted(trades['account'].astype(str).unique().tolist())
selected_accounts = st.sidebar.multiselect('Filter accounts (sample)', options=accounts, default=None)

symbols = None
if 'symbol' in trades.columns:
    symbols = sorted(trades['symbol'].astype(str).unique().tolist())
    selected_symbols = st.sidebar.multiselect('Symbols', options=symbols, default=None)
else:
    selected_symbols = None

sentiments_available = sorted(daily['sentiment'].dropna().unique().tolist())
selected_sent = st.sidebar.multiselect('Sentiment', options=sentiments_available, default=sentiments_available)

leverage_max = None
if 'leverage_mean' in daily.columns:
    leverage_max = float(np.nanmax(daily['leverage_mean']))
    lev_cut = st.sidebar.slider('Max avg leverage (daily)', min_value=0.0, max_value=max(1.0, leverage_max), value=max(1.0, leverage_max))
else:
    lev_cut = None

# Apply filters
mask = (pd.to_datetime(daily['Date']).dt.date >= date_range[0]) & (pd.to_datetime(daily['Date']).dt.date <= date_range[1])
if selected_sent:
    mask &= daily['sentiment'].isin(selected_sent)
if lev_cut is not None and 'leverage_mean' in daily.columns:
    mask &= daily['leverage_mean'] <= lev_cut

filtered_daily = daily[mask].copy()

# For trade-level filters (accounts/symbol)
trade_mask = (pd.to_datetime(trades['time_dt']).dt.date >= date_range[0]) & (pd.to_datetime(trades['time_dt']).dt.date <= date_range[1])
if selected_accounts:
    trade_mask &= trades['account'].astype(str).isin(selected_accounts)
if selected_symbols:
    trade_mask &= trades['symbol'].astype(str).isin(selected_symbols)

filtered_trades = trades[trade_mask].copy()

# ---------------------- Layout ----------------------
st.title('Trader Performance × Bitcoin Fear & Greed')
st.markdown('Interactive dashboard to explore how market sentiment relates to trader performance.')

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Period start', str(date_range[0]))
with col2:
    st.metric('Period end', str(date_range[1]))
with col3:
    st.metric('Days shown', int(filtered_daily.shape[0]))
with col4:
    st.metric('Unique accounts (period)', int(filtered_trades['account'].nunique()))

# Main charts
st.header('Overview')
left, right = st.columns([2,1])

with left:
    st.subheader('Daily PnL (sum)')
    fig_pnl = px.line(filtered_daily, x='Date', y='closedPnL_sum', title='Daily closed PnL (sum)')
    fig_pnl.update_traces(mode='lines+markers')
    st.plotly_chart(fig_pnl, use_container_width=True)

    st.subheader('Win rate (daily)')
    fig_wr = px.line(filtered_daily, x='Date', y='win_rate', title='Daily win rate')
    st.plotly_chart(fig_wr, use_container_width=True)

with right:
    st.subheader('Sentiment counts (period)')
    sent_counts = filtered_daily['sentiment'].value_counts().reset_index()
    sent_counts.columns = ['sentiment','count']
    fig_sent = px.pie(sent_counts, names='sentiment', values='count', title='Sentiment distribution')
    st.plotly_chart(fig_sent, use_container_width=True)

# Sentiment impact
st.header('Sentiment Impact')
colA, colB = st.columns(2)
with colA:
    st.subheader('PnL by Sentiment')
    box = px.box(filtered_daily, x='sentiment', y='closedPnL_mean', points='all', title='Daily mean closedPnL by sentiment')
    st.plotly_chart(box, use_container_width=True)
with colB:
    st.subheader('Win rate by Sentiment')
    box2 = px.box(filtered_daily, x='sentiment', y='win_rate', points='all', title='Win rate by sentiment')
    st.plotly_chart(box2, use_container_width=True)

# Per-account exploration
st.header('Account-level (sample)')
if selected_accounts:
    account_to_show = st.selectbox('Choose account to inspect', options=selected_accounts)
    acct_trades = trades[trades['account'].astype(str) == str(account_to_show)].copy()
    acct_trades = acct_trades[(pd.to_datetime(acct_trades['time_dt']).dt.date >= date_range[0]) & (pd.to_datetime(acct_trades['time_dt']).dt.date <= date_range[1])]
    st.write(f'Account {account_to_show} — trades shown: {len(acct_trades)}')
    if len(acct_trades) > 0:
        fig_acct = px.histogram(acct_trades, x='closedPnL', nbins=80, title=f'PNL distribution — account {account_to_show}')
        st.plotly_chart(fig_acct, use_container_width=True)
        st.dataframe(acct_trades.head(200))
    else:
        st.info('No trades for this account in selected period.')
else:
    st.info('Select one or more accounts in the sidebar to enable account-level analysis.')

# Statistical tests
st.header('Statistical tests')
st.write('Compare Fear vs Greed for daily mean closedPnL and win_rate (period filter applied).')
if st.button('Run Fear vs Greed tests'):
    fear = filtered_daily[filtered_daily['sentiment']=='Fear']['closedPnL_mean'].dropna()
    greed = filtered_daily[filtered_daily['sentiment']=='Greed']['closedPnL_mean'].dropna()
    out = {}
    if len(fear)>1 and len(greed)>1:
        out['closedPnL_ttest'] = ttest_ind(fear, greed, equal_var=False)
        out['closedPnL_mannwhitney'] = mannwhitneyu(fear, greed)
    else:
        out['closedPnL'] = 'Not enough data for Fear vs Greed'

    fear_wr = filtered_daily[filtered_daily['sentiment']=='Fear']['win_rate'].dropna()
    greed_wr = filtered_daily[filtered_daily['sentiment']=='Greed']['win_rate'].dropna()
    if len(fear_wr)>1 and len(greed_wr)>1:
        out['winrate_ttest'] = ttest_ind(fear_wr, greed_wr, equal_var=False)
        out['winrate_mannwhitney'] = mannwhitneyu(fear_wr, greed_wr)
    else:
        out['winrate'] = 'Not enough data for Fear vs Greed'

    st.write(out)

# Regression quick model
st.header('Quick regression (daily)')
if st.button('Run OLS: closedPnL_mean ~ sentiment + leverage_mean + account_count'):
    reg_df = filtered_daily.copy()
    acct_col = [c for c in reg_df.columns if 'account' in c.lower() or 'nunique' in c.lower()]
    if acct_col:
        reg_df = reg_df.rename(columns={acct_col[0]:'account_count'})
    reg_df = reg_df.dropna(subset=['closedPnL_mean'])
    if reg_df.shape[0] < 5:
        st.warning('Not enough rows to fit regression')
    else:
        covars = ['C(sentiment)']
        if 'leverage_mean' in reg_df.columns:
            covars.append('leverage_mean')
        if 'account_count' in reg_df.columns:
            covars.append('account_count')
        formula = 'closedPnL_mean ~ ' + ' + '.join(covars)
        try:
            model = smf.ols(formula=formula, data=reg_df).fit()
            st.text(model.summary().as_text())
        except Exception as e:
            st.error(f'Regression error: {e}')

# Download aggregated data
st.header('Download data')
st.write('Download the cleaned daily aggregated CSV for offline analysis.')
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_daily)
st.download_button('Download daily_aggregated.csv', data=csv, file_name='daily_aggregated.csv', mime='text/csv')

st.markdown('---')
st.caption('Built with ❤️ — choose "Advanced" if you want more tabs, controls, and cohort analysis.')
