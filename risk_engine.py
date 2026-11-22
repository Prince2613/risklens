import numpy as np
import pandas as pd


def compute_metrics(prices, weights=None, benchmark=None, rf=0.06):
    returns = prices.pct_change().dropna()

    if weights is None:
        weights = np.array([1 / len(returns.columns)] * len(returns.columns))
    weights = np.array(weights, dtype=float) / np.sum(weights)

    port_returns = returns.dot(weights)

    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0

    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()

    var_95 = np.percentile(port_returns, 5)
    es_95 = port_returns[port_returns <= var_95].mean()

    corr = returns.corr()

    cov = returns.cov() * 252
    port_var = weights @ cov.values @ weights
    port_vol = np.sqrt(port_var)
    marginal = cov.values @ weights
    risk_contrib = weights * marginal / port_vol if port_vol > 0 else weights * 0

    beta = None
    if benchmark is not None:
        bench_returns = benchmark.pct_change().dropna()
        aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
        if len(aligned) > 30:
            cov_pb = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            beta = cov_pb[0, 1] / cov_pb[1, 1]

    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'var_95': var_95,
        'es_95': es_95,
        'beta': beta,
        'corr': corr,
        'risk_contrib': dict(zip(returns.columns, risk_contrib)),
    }


def risk_level(ann_vol):
    if ann_vol < 0.15: return 'Low'
    if ann_vol < 0.25: return 'Moderate'
    if ann_vol < 0.40: return 'High'
    return 'Severe'


def get_recommendation(ann_vol, risk_lvl, target_vol=0.15):
    """Vol-targeting position sizing + plain-English action."""
    if ann_vol <= 0:
        suggested_alloc_pct = 100
    else:
        suggested_alloc_pct = round(min(target_vol / ann_vol, 1.0) * 100)

    messages = {
        'Low': f"Risk is low. Full allocation is reasonable at current volatility.",
        'Moderate': f"Risk is moderate. Sizing near {suggested_alloc_pct}% of intended capital keeps portfolio volatility near a {int(target_vol*100)}% annual target.",
        'High': f"Risk is elevated. Consider trimming to roughly {suggested_alloc_pct}% of intended capital, or hedge with protective puts.",
        'Severe': f"Risk is severe. Consider reducing exposure to roughly {suggested_alloc_pct}% of intended capital, or moving to cash until volatility subsides.",
    }

    return {
        'suggested_allocation_pct': suggested_alloc_pct,
        'action_text': messages.get(risk_lvl, messages['Moderate']),
    }