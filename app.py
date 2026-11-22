from flask import Flask, render_template, request, jsonify
from flask_login import LoginManager, login_required, current_user
from models import db, User
from auth import auth
from data_fetcher import fetch_stock, fetch_multiple
from risk_engine import compute_metrics, risk_level, get_recommendation
from stress_test import compute_stress_test
import plotly.graph_objects as go
import plotly.utils
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///risklens.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))


app.register_blueprint(auth)


@app.after_request
def add_no_cache(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ---------- Pages ----------

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name)


# ---------- Ticker list for search ----------

TICKER_LIST = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
    {"symbol": "ITC.NS", "name": "ITC Limited"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
    {"symbol": "WIPRO.NS", "name": "Wipro"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma"},
    {"symbol": "TITAN.NS", "name": "Titan Company"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises"},
    {"symbol": "^NSEI", "name": "NIFTY 50 Index"},
    {"symbol": "^BSESN", "name": "BSE SENSEX"},
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "GOOGL", "name": "Alphabet (Google)"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "META", "name": "Meta Platforms"},
    {"symbol": "NFLX", "name": "Netflix"},
    {"symbol": "JPM", "name": "JPMorgan Chase"},
    {"symbol": "V", "name": "Visa"},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "DIS", "name": "Walt Disney"},
    {"symbol": "BA", "name": "Boeing"},
    {"symbol": "SPY", "name": "S&P 500 ETF"},
    {"symbol": "QQQ", "name": "Nasdaq 100 ETF"},
]


@app.route('/api/tickers')
def api_tickers():
    return jsonify(TICKER_LIST)


# ---------- Range mapping ----------

RANGE_MAP = {
    '1D': ('1d', '5m'),
    '5D': ('5d', '15m'),
    '1M': ('1mo', '1d'),
    '3M': ('3mo', '1d'),
    'YTD': ('ytd', '1d'),
    '1Y': ('1y', '1d'),
    '5Y': ('5y', '1wk'),
    'ALL': ('max', '1mo'),
}


def _resolve_range(range_key):
    return RANGE_MAP.get(range_key, RANGE_MAP['1Y'])


# ---------- Chart builder ----------

def _chart_x(df):
    """ISO dates Plotly can parse. Timezone suffixes like +05:30 break candlesticks."""
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    hours = getattr(idx, 'hour', None)
    mins = getattr(idx, 'minute', None)
    is_daily = True
    if hours is not None and len(hours):
        is_daily = int(max(hours)) == 0 and int(max(mins)) == 0
    fmt = '%Y-%m-%d' if is_daily else '%Y-%m-%d %H:%M:%S'
    return [ts.strftime(fmt) for ts in idx]


def _ohlc_lists(df):
    return (
        _chart_x(df),
        df['Open'].astype(float).tolist(),
        df['High'].astype(float).tolist(),
        df['Low'].astype(float).tolist(),
        df['Close'].astype(float).tolist(),
    )


def _build_chart(df, ticker, chart_type='candle', height=420):
    print(f"[_build_chart] {ticker} rows={len(df)} first={df.index[0]} last={df.index[-1]} "
          f"lastClose={df['Close'].iloc[-1]}")

    x, o, h, l, c = _ohlc_lists(df)

    if chart_type == 'candle':
        trace = go.Candlestick(
            x=x, open=o, high=h, low=l, close=c,
            name=ticker,
            increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
            decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350'),
        )
        fig = go.Figure(data=[trace])
    elif chart_type == 'line':
        trace = go.Scatter(
            x=x, y=c,
            mode='lines', name=ticker,
            line=dict(color='#4f9cf9', width=1.5),
        )
        fig = go.Figure(data=[trace])
    elif chart_type == 'bar':
        trace = go.Ohlc(
            x=x, open=o, high=h, low=l, close=c,
            name=ticker,
            increasing=dict(line=dict(color='#26a69a')),
            decreasing=dict(line=dict(color='#ef5350')),
        )
        fig = go.Figure(data=[trace])
    elif chart_type == 'area':
        trace = go.Scatter(
            x=x, y=c,
            mode='lines', name=ticker,
            line=dict(color='#4f9cf9', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(79,156,249,0.15)',
        )
        fig = go.Figure(data=[trace])
    else:
        return None

    fig.update_layout(
        paper_bgcolor='#131721',
        plot_bgcolor='#131721',
        font=dict(color='#e6edf3', family='Inter'),
        margin=dict(l=40, r=20, t=20, b=40),
        height=height,
        hovermode='x unified',
        dragmode='pan',
        xaxis=dict(
            gridcolor='#1f2633',
            type='date',
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
        ),
        yaxis=dict(gridcolor='#1f2633', showspikes=True),
    )
    fig_json = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

    first = df.iloc[0]
    last = df.iloc[-1]
    change = float(last['Close']) - float(first['Open'])
    change_pct = (change / float(first['Open']) * 100) if first['Open'] else 0.0

    ohlc = {
        'open': round(float(last['Open']), 2),
        'high': round(float(df['High'].max()), 2),
        'low': round(float(df['Low'].min()), 2),
        'close': round(float(last['Close']), 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 2),
        'date': str(df.index[-1]),
    }

    return {'figure': fig_json, 'ohlc': ohlc}


# ---------- Charts ----------

@app.route('/api/chart-public/<ticker>')
def chart_public(ticker):
    chart_type = request.args.get('type', 'candle')
    range_key = request.args.get('range', '6M')
    if range_key == '6M':
        period, interval = ('6mo', '1d')
    else:
        period, interval = _resolve_range(range_key)
    df = fetch_stock(ticker, period=period, interval=interval)
    if df is None:
        print(f"[chart_public] {ticker} -> NO DATA")
        return jsonify({'error': 'Ticker not found'}), 404
    result = _build_chart(df, ticker, chart_type, height=340)
    if result is None:
        return jsonify({'error': 'Chart build failed'}), 500
    return jsonify(result)


@app.route('/api/chart/<ticker>')
@login_required
def chart(ticker):
    chart_type = request.args.get('type', 'candle')
    range_key = request.args.get('range', '1Y')
    period, interval = _resolve_range(range_key)
    df = fetch_stock(ticker, period=period, interval=interval)
    if df is None:
        print(f"[chart] {ticker} -> NO DATA")
        return jsonify({'error': 'Ticker not found'}), 404
    result = _build_chart(df, ticker, chart_type, height=420)
    if result is None:
        return jsonify({'error': 'Chart build failed'}), 500
    return jsonify(result)


# ---------- Helpers ----------

def _parse_portfolio(raw):
    tickers, weights = [], []
    for part in raw.split(','):
        part = part.strip()
        if ':' in part:
            try:
                t, w = part.split(':')
                tickers.append(t.strip().upper())
                weights.append(float(w))
            except ValueError:
                continue
    return tickers, weights


# ---------- Portfolio analysis ----------

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.get_json()
    raw = data.get('portfolio', '')
    portfolio_value = float(data.get('value', 100000))

    print(f"[analyze] raw input: {raw!r}")

    tickers, weights = _parse_portfolio(raw)
    print(f"[analyze] parsed tickers={tickers} weights={weights}")

    if len(tickers) < 2:
        print("[analyze] FAILED: fewer than 2 tickers parsed")
        return jsonify({'error': 'Need at least 2 valid stocks (format: TICKER:WEIGHT)'}), 400

    prices = fetch_multiple(tickers, period='1y')
    print(f"[analyze] fetched columns: {list(prices.columns)}")

    if prices.empty or len(prices.columns) < 2:
        failed = [t for t in tickers if t not in prices.columns]
        print(f"[analyze] FAILED: empty={prices.empty} cols={list(prices.columns)} missing={failed}")
        if failed:
            return jsonify({'error': f'Could not fetch data for: {", ".join(failed)}'}), 400
        return jsonify({'error': 'No overlapping trading days for these tickers'}), 400

    valid_tickers = list(prices.columns)
    valid_weights = [weights[tickers.index(t)] for t in valid_tickers]

    metrics = compute_metrics(prices, valid_weights)
    lvl = risk_level(metrics['ann_vol'])
    recommendation = get_recommendation(metrics['ann_vol'], lvl)

    eq_metrics = compute_metrics(prices, None)

    result = {
        'ann_return': round(metrics['ann_return'] * 100, 2),
        'ann_vol': round(metrics['ann_vol'] * 100, 2),
        'sharpe': round(metrics['sharpe'], 2),
        'max_dd': round(metrics['max_dd'] * 100, 2),
        'var_95': round(metrics['var_95'] * 100, 2),
        'es_95': round(metrics['es_95'] * 100, 2),
        'beta': round(metrics['beta'], 2) if metrics['beta'] else None,
        'risk_level': lvl,
        'var_loss': round(abs(metrics['var_95']) * portfolio_value, 0),
        'es_loss': round(abs(metrics['es_95']) * portfolio_value, 0),
        'dd_loss': round(abs(metrics['max_dd']) * portfolio_value, 0),
        'risk_contrib': {k: round(v * 100, 2) for k, v in metrics['risk_contrib'].items()},
        'corr': metrics['corr'].round(2).to_dict(),
        'recommendation': recommendation,
        'comparison': {
            'your_portfolio': {
                'return': round(metrics['ann_return'] * 100, 2),
                'vol': round(metrics['ann_vol'] * 100, 2),
                'sharpe': round(metrics['sharpe'], 2),
            },
            'equal_weight': {
                'return': round(eq_metrics['ann_return'] * 100, 2),
                'vol': round(eq_metrics['ann_vol'] * 100, 2),
                'sharpe': round(eq_metrics['sharpe'], 2),
            },
        },
        'valid_tickers': valid_tickers,
    }
    return jsonify(result)


# ---------- Stress test ----------

@app.route('/api/stress', methods=['POST'])
@login_required
def stress():
    data = request.get_json()
    raw = data.get('portfolio', '')
    tickers, weights = _parse_portfolio(raw)

    if len(tickers) < 2:
        return jsonify({'error': 'Need at least 2 valid stocks'}), 400

    results = compute_stress_test(tickers, weights)
    return jsonify({'scenarios': results})


# ---------- Run ----------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)