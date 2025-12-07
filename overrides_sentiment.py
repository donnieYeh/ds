import json


def compute_dynamic_base_usdt(exchange, symbol, leverage, contract_size, min_amount, fallback_base_usdt, safety_ratio=0.8):
    """Return a nominal base (USDT) sized to ensure min contracts are tradable.

    Uses equity (total>free+used>free) for budget, then clamps by safety_ratio×leverage.
    """
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {}) or {}
        free = float(usdt.get('free', 0) or 0)
        used = float(usdt.get('used', 0) or 0)
        total = float(usdt.get('total', 0) or 0)
        equity = total if total > 0 else (free + used if free + used > 0 else free)
    except Exception:
        equity = 0.0

    max_budget = equity * float(safety_ratio or 1) * float(leverage or 1)

    price = None
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get('last') or ticker.get('close')
        price = float(price) if price is not None else None
    except Exception:
        price = None

    if not price or not contract_size or not min_amount:
        return min(float(fallback_base_usdt), float(max_budget)) if max_budget > 0 else 0.0

    min_notional = float(price) * float(contract_size) * float(min_amount)
    target = max(float(fallback_base_usdt), float(min_notional))
    base_usdt = min(target, float(max_budget))

    if base_usdt < min_notional:
        return 0.0

    return float(base_usdt)


def get_equity_info(exchange):
    """Return USDT free/used/total and chosen equity (float)."""
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get('USDT', {}) or {}
        free = float(usdt.get('free', 0) or 0)
        used = float(usdt.get('used', 0) or 0)
        total = float(usdt.get('total', 0) or 0)
        equity = total if total > 0 else (free + used if free + used > 0 else free)
        return {"free": free, "used": used, "total": total, "equity": equity}
    except Exception:
        return {"free": 0.0, "used": 0.0, "total": 0.0, "equity": 0.0}


def compute_nominal_budget(equity, leverage, safety_ratio):
    try:
        return float(equity) * float(leverage or 1) * float(safety_ratio or 1)
    except Exception:
        return 0.0


def compute_min_notional(price, contract_size, min_amount):
    try:
        return float(price) * float(contract_size) * float(min_amount)
    except Exception:
        return 0.0


def compute_atr_stop_distance(df, period=14, multiple=1.5):
    """ATR-based stop distance; fallback to 0.5% of last close."""
    try:
        import pandas as _pd  # ensure pandas ops available
        if df is None or len(df) < period + 1:
            raise ValueError("insufficient df")
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        closes = df['close'].astype(float)
        prev_close = closes.shift(1)
        tr1 = (highs - lows).abs()
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)
        atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
        if not atr or atr <= 0:
            raise ValueError("atr invalid")
        return float(atr) * float(multiple)
    except Exception:
        try:
            last_close = float(df['close'].iloc[-1]) if df is not None else None
        except Exception:
            last_close = None
        return (last_close * 0.005) if last_close else 1.0


def compute_risk_based_contracts(R_usdt, stop_distance, contract_size):
    try:
        return max(0.0, float(R_usdt) / (float(stop_distance) * float(contract_size)))
    except Exception:
        return 0.0


def pretrade_feasible_contracts(exchange, symbol, contracts, price, contract_size, leverage, free_usdt, taker_fee_rate=0.0005, cushion=1.02):
    """Down-adjust contracts so margin+fees fit into free USDT; return rounded contracts."""
    try:
        contracts = float(contracts)
        notional = float(price) * float(contract_size) * contracts
        required_margin = notional / float(leverage or 1)
        fees = notional * float(taker_fee_rate)
        need = (required_margin + fees) * float(cushion)
        if need <= float(free_usdt):
            return round(contracts, 2)
        scale = max(0.0, float(free_usdt) / need)
        return round(contracts * scale, 2)
    except Exception:
        return round(float(contracts), 2)


def print_raw_positions(exchange, symbol):
    try:
        positions = exchange.fetch_positions([symbol])
        print("[OKX 原生持仓数据] =>")
        try:
            print(json.dumps(positions, ensure_ascii=False, indent=2))
        except Exception:
            print(str(positions))
    except Exception as e:
        print(f"[OKX 原生持仓数据] 获取失败: {e}")


# ===== Coin override helpers =====
_ASSET_CODE = "ETH"


def get_asset_code() -> str:
    return _ASSET_CODE


def get_asset_symbol() -> str:
    # Default to USDT-margined perpetual symbol format used by OKX in this repo
    return f"{_ASSET_CODE}/USDT:USDT"


def get_human_pair() -> str:
    return f"{_ASSET_CODE}/USDT"


def get_sentiment_tokens():
    # Cryptoracle API expects tokens like ["BTC"]
    return [get_asset_code()]


def get_price_label() -> str:
    return f"{get_asset_code()}当前价格"


def get_contract_unit_name() -> str:
    # For messages like: 1张 = X <unit>
    return get_asset_code()
