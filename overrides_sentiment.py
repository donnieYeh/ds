import json


def compute_dynamic_base_usdt(exchange, symbol, leverage, contract_size, min_amount, fallback_base_usdt, safety_ratio=0.8):
    try:
        balance = exchange.fetch_balance()
        usdt_free = float(balance.get('USDT', {}).get('free', 0) or 0)
    except Exception:
        usdt_free = 0.0

    max_budget = usdt_free * safety_ratio * float(leverage or 1)

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
