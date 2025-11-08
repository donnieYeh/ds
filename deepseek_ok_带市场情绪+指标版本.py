import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import re
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
from overrides_sentiment import (
    compute_dynamic_base_usdt,
    get_equity_info,
    compute_nominal_budget,
    compute_min_notional,
    compute_atr_stop_distance,
    compute_risk_based_contracts,
    pretrade_feasible_contracts,
    get_asset_code,
    get_asset_symbol,
    get_human_pair,
    get_sentiment_tokens,
    get_price_label,
    get_contract_unit_name,
)

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 读取环境变量中可配置的最近K线数量，默认20，限定范围1-200
def _get_recent_kline_count_default() -> int:
    try:
        val = int(os.getenv('RECENT_KLINE_COUNT', '20'))
        return max(1, min(200, val))
    except Exception:
        return 20

# 解析布尔类环境变量（"1/true/yes/on" 为真）
def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
})

# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'symbol': get_asset_symbol(),  # 由外部override提供
    'leverage': 10,  # 杠杆倍数,只影响保证金不影响下单价值
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
    'require_high_confidence_entry': _get_bool_env('REQUIRE_HIGH_CONFIDENCE_ENTRY', True),  # 是否仅允许高信心开单
    'data_points': 96,  # 24小时数据（96根15分钟K线）
    'recent_kline_count': _get_recent_kline_count_default(),  # 近N根K线用于提示/决策
    'print_prompt': _get_bool_env('PRINT_PROMPT', False),  # 是否打印提示词
    'analysis_periods': {
        'short_term': 20,  # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96  # 长期趋势
    },
    # 新增智能仓位参数
    'position_management': {
        'enable_intelligent_position': True,  # 🆕 新增：是否启用智能仓位管理
        'base_usdt_amount': 100,  # USDT投入下单基数
        'high_confidence_multiplier': 1.5,
        'medium_confidence_multiplier': 1.0,
        'low_confidence_multiplier': 0.5,
        'max_position_ratio': 10,  # 单次最大仓位比例
        'trend_strength_multiplier': 1.2
    }
}


def print_runtime_config():
    """启动时打印关键可配置项（不含敏感信息）。"""
    try:
        cfg = TRADE_CONFIG
        ap = cfg.get('analysis_periods', {})
        pm = cfg.get('position_management', {})
        env_recent = os.getenv('RECENT_KLINE_COUNT')
        env_print_prompt = os.getenv('PRINT_PROMPT')

        print("\n【运行配置】")
        print(f"- 交易对: {get_human_pair()} ({cfg.get('symbol')})")
        print(f"- 周期: {cfg.get('timeframe')}  杠杆: {cfg.get('leverage')}x  模式: {'测试' if cfg.get('test_mode') else '实盘'}")
        print(f"- 历史K线数量(data_points): {cfg.get('data_points')}")
        recent_line = f"- 最近K线数量(recent_kline_count): {cfg.get('recent_kline_count')}"
        if env_recent:
            recent_line += f"  (来自环境变量 RECENT_KLINE_COUNT={env_recent})"
        print(recent_line)
        print(
            f"- 打印Prompt: {'启用' if cfg.get('print_prompt') else '禁用'}"
            + (f"  (来自环境变量 PRINT_PROMPT={env_print_prompt})" if env_print_prompt is not None else "")
        )
        print(f"- 指标周期: 短期={ap.get('short_term')}, 中期={ap.get('medium_term')}, 长期={ap.get('long_term')}")
        print(
            "- 智能仓位: "
            + ("启用" if pm.get('enable_intelligent_position', True) else "禁用")
            + f"; 基数USDT={pm.get('base_usdt_amount')}, 倍数(H/M/L)="
            + f"{pm.get('high_confidence_multiplier')}/{pm.get('medium_confidence_multiplier')}/{pm.get('low_confidence_multiplier')}, "
            + f"最大仓位比例={pm.get('max_position_ratio')}, 趋势倍数={pm.get('trend_strength_multiplier')}"
        )
        require_high = cfg.get('require_high_confidence_entry', True)
        env_require_high = os.getenv('REQUIRE_HIGH_CONFIDENCE_ENTRY')
        print(
            f"- 高信心开单限制: {'启用' if require_high else '禁用'}"
            + (f"  (来自环境变量 REQUIRE_HIGH_CONFIDENCE_ENTRY={env_require_high})" if env_require_high is not None else "")
        )
    except Exception as e:
        print(f"⚠️ 配置打印失败: {e}")


def setup_exchange():
    """设置交易所参数 - 强制全仓模式"""
    try:

        # 首先获取合约规格信息
        print(f"🔍 获取{get_asset_code()}合约规格...")
        markets = exchange.load_markets()
        btc_market = markets[TRADE_CONFIG['symbol']]

        # 获取合约乘数
        contract_size = float(btc_market['contractSize'])
        print(f"✅ 合约规格: 1张 = {contract_size} {get_contract_unit_name()}")

        # 存储合约规格到全局配置
        TRADE_CONFIG['contract_size'] = contract_size
        TRADE_CONFIG['min_amount'] = btc_market['limits']['amount']['min']

        print(f"📏 最小交易量: {TRADE_CONFIG['min_amount']} 张")

        # 先检查现有持仓
        print("🔍 检查现有持仓模式...")
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        has_isolated_position = False
        isolated_position_info = None

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos.get('contracts', 0))
                mode = pos.get('mgnMode')

                if contracts > 0 and mode == 'isolated':
                    has_isolated_position = True
                    isolated_position_info = {
                        'side': pos.get('side'),
                        'size': contracts,
                        'entry_price': pos.get('entryPrice'),
                        'mode': mode
                    }
                    break

        # 2. 如果有逐仓持仓，提示并退出
        if has_isolated_position:
            print("❌ 检测到逐仓持仓，程序无法继续运行！")
            print(f"📊 逐仓持仓详情:")
            print(f"   - 方向: {isolated_position_info['side']}")
            print(f"   - 数量: {isolated_position_info['size']}")
            print(f"   - 入场价: {isolated_position_info['entry_price']}")
            print(f"   - 模式: {isolated_position_info['mode']}")
            print("\n🚨 解决方案:")
            print("1. 手动平掉所有逐仓持仓")
            print("2. 或者将逐仓持仓转为全仓模式")
            print("3. 然后重新启动程序")
            return False

        # 3. 设置单向持仓模式
        print("🔄 设置单向持仓模式...")
        try:
            exchange.set_position_mode(False, TRADE_CONFIG['symbol'])  # False表示单向持仓
            print("✅ 已设置单向持仓模式")
        except Exception as e:
            print(f"⚠️ 设置单向持仓模式失败 (可能已设置): {e}")

        # 4. 设置全仓模式和杠杆
        print("⚙️ 设置全仓模式和杠杆...")
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 强制全仓模式
        )
        print(f"✅ 已设置全仓模式，杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 5. 验证设置
        print("🔍 验证账户设置...")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"💰 当前USDT余额: {usdt_balance:.2f}")

        # 获取当前持仓状态
        current_pos = get_current_position()
        if current_pos:
            print(f"📦 当前持仓: {current_pos['side']}仓 {current_pos['size']}张")
        else:
            print("📦 当前无持仓")

        print("🎯 程序配置完成：全仓模式 + 单向持仓")
        return True

    except Exception as e:
        print(f"❌ 交易所设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

# 反手平仓事件位图（低位为最近一次），用于限频
# 注意：必须在每次评估周期都左移一次（无反手则写入0，有反手则写入1），
# 否则会因为只在反手时记录而永久保持为1，导致误判“近期有反手”。
reduce_hist = 0


def _can_reverse_recently() -> bool:
    """最近3次无反手平仓事件时才允许反手。"""
    mask = 0b111
    return (reduce_hist & mask) == 0


def _record_reverse_close_event(did_reverse: bool = True):
    """记录一次评估周期的反手事件：
    - did_reverse=True：左移并置1，表示本周期发生了反手平仓
    - did_reverse=False：左移并置0，表示本周期未发生反手平仓
    保持8位窗口。
    """
    global reduce_hist
    reduce_hist = ((reduce_hist << 1) | (1 if did_reverse else 0)) & 0xFF


def calculate_intelligent_position_v2(signal_data, price_data, current_position):
    """智能仓位（权益预算 + ATR风险 + 可行性 + 同向不减仓）"""
    config = TRADE_CONFIG['position_management']
    if not config.get('enable_intelligent_position', True):
        return 0.1

    try:
        eq = get_equity_info(exchange)
        usdt_free = eq['free']
        equity = eq['equity']
        safety_ratio = config.get('safety_ratio', 0.8)

        base = compute_dynamic_base_usdt(
            exchange,
            TRADE_CONFIG['symbol'],
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG.get('contract_size', 0.01),
            TRADE_CONFIG.get('min_amount', 0.01),
            config['base_usdt_amount'],
            safety_ratio,
        ) or config['base_usdt_amount']

        budget = compute_nominal_budget(equity, TRADE_CONFIG['leverage'], safety_ratio)
        conf_mult = {
            'HIGH': config['high_confidence_multiplier'],
            'MEDIUM': config['medium_confidence_multiplier'],
            'LOW': config['low_confidence_multiplier']
        }.get(signal_data.get('confidence'), 1.0)
        trend = price_data['trend_analysis'].get('overall', '震荡整理')
        trend_mult = config['trend_strength_multiplier'] if trend in ['强势上涨', '强势下跌'] else 1.0
        rsi = price_data['technical_data'].get('rsi', 50)
        rsi_mult = 0.7 if (rsi > 75 or rsi < 25) else 1.0

        suggested = base * conf_mult * trend_mult * rsi_mult
        policy_cap = equity * config.get('max_position_ratio', 10)
        final_nominal = min(suggested, budget, policy_cap)
        nominal_contracts = final_nominal / (price_data['price'] * TRADE_CONFIG['contract_size'])

        stop_dist = compute_atr_stop_distance(price_data.get('full_data'), config.get('atr_period', 14), config.get('atr_multiple', 1.5))
        R_usdt = equity * config.get('risk_per_trade_ratio', 0.01)
        risk_contracts = compute_risk_based_contracts(R_usdt, stop_dist, TRADE_CONFIG['contract_size'])

        target = round(min(nominal_contracts, risk_contracts), 2)
        feasible = pretrade_feasible_contracts(
            exchange,
            TRADE_CONFIG['symbol'],
            target,
            price_data['price'],
            TRADE_CONFIG['contract_size'],
            TRADE_CONFIG['leverage'],
            usdt_free,
            config.get('taker_fee_rate', 0.0005),
            1.02,
        )

        min_ct = TRADE_CONFIG.get('min_amount', 0.01)
        signal_side = 'long' if signal_data.get('signal') == 'BUY' else ('short' if signal_data.get('signal') == 'SELL' else None)
        if feasible is None or feasible <= 0:
            return 0
        if 0 < feasible < min_ct:
            if current_position and signal_side and current_position.get('side') == signal_side:
                feasible = current_position.get('size', min_ct)
            else:
                return 0

        if current_position and signal_side and current_position.get('side') == signal_side:
            if feasible < current_position.get('size', 0):
                feasible = current_position['size']

        return round(feasible, 2)
    except Exception:
        # fallback: fixed tiny contract
        return max(TRADE_CONFIG.get('min_amount', 0.01), 0.01)


def generate_sma_analysis(source, short=5, mid=20, long=80, price_col="close"):
    """
    基于已计算好的 5 / 20 / 80 周期 SMA 生成面向 LLM 的趋势描述文本。

    支持两种输入:
        - price_data 字典：需包含 'full_data' (带有 sma_X 列) 与当前 price
        - DataFrame：需包含 close 及相应的 sma_X 列
    """
    import numpy as np

    price_now = None
    df = None
    tech = {}

    if isinstance(source, dict):
        price_data = source
        df = price_data.get('full_data')
        tech = price_data.get('technical_data', {})
        price_now = price_data.get('price')
    else:
        df = source

    if df is None or len(df) < long + 5:
        return "📈 移动平均线分析：数据不足，暂无法给出可靠的均线趋势评估，仅供参考。"

    sma_cols = {
        'short': f'sma_{short}',
        'mid': f'sma_{mid}',
        'long': f'sma_{long}'
    }

    for col in sma_cols.values():
        if col not in df.columns:
            return f"📈 移动平均线分析：缺少 {col} 数据，暂无法评估均线结构。"

    sma_s = df[sma_cols['short']].astype(float)
    sma_m = df[sma_cols['mid']].astype(float)
    sma_l = df[sma_cols['long']].astype(float)

    price_series = df[price_col].astype(float) if price_col in df.columns else None
    if price_now is None and price_series is not None:
        price_now = float(price_series.iloc[-1])
    elif price_now is None:
        return "📈 移动平均线分析：缺少价格数据，无法完成评估。"

    sma_s_now = float(tech.get(sma_cols['short'], sma_s.iloc[-1])) if tech else float(sma_s.iloc[-1])
    sma_m_now = float(tech.get(sma_cols['mid'], sma_m.iloc[-1])) if tech else float(sma_m.iloc[-1])
    sma_l_now = float(tech.get(sma_cols['long'], sma_l.iloc[-1])) if tech else float(sma_l.iloc[-1])
    price_now = float(price_now)

    # 如有 NaN，直接降级提示
    if any(np.isnan([sma_s_now, sma_m_now, sma_l_now])):
        return "📈 移动平均线分析：当前均线数据尚未完全形成，暂不作为主要决策依据。"

    # 均线结构判定
    if sma_s_now > sma_m_now > sma_l_now:
        structure = "5 > 20 > 80，形成多头排列，趋势偏多。"
    elif sma_s_now < sma_m_now < sma_l_now:
        structure = "5 < 20 < 80，形成空头排列，趋势偏空。"
    else:
        structure = "均线互相纠缠或缺乏明确排列结构，偏震荡或趋势不明。"

    # 价格相对位置
    max_sma = max(sma_s_now, sma_m_now, sma_l_now)
    min_sma = min(sma_s_now, sma_m_now, sma_l_now)

    if price_now > max_sma:
        pos_desc = "当前价格位于所有均线上方，属相对强势区域，偏多头环境。"
    elif price_now < min_sma:
        pos_desc = "当前价格位于所有均线下方，属相对弱势区域，偏空头环境。"
    else:
        # 介于某些均线之间，给一点层次感
        if price_now >= sma_m_now:
            pos_desc = "当前价格介于中长期均线附近，短期虽有支撑，但上方仍需观察动能延续。"
        elif price_now <= sma_m_now:
            pos_desc = "当前价格介于短中均线之间，存在震荡或方向选择阶段。"
        else:
            pos_desc = "当前价格位于均线密集区附近，市场处于震荡平衡状态。"

    # 趋势稳定性：看均线斜率是否同向
    def slope(series, window=3):
        if len(series.dropna()) < window + 1:
            return 0.0
        return float(series.iloc[-1] - series.iloc[-1 - window])

    slope_s = slope(sma_s)
    slope_m = slope(sma_m)
    slope_l = slope(sma_l)

    same_direction = (slope_s >= 0 and slope_m >= 0 and slope_l >= 0) or \
                     (slope_s <= 0 and slope_m <= 0 and slope_l <= 0)

    if same_direction and abs(slope_l) > 0:
        stability = "短中长周期均线大致同向，趋势具有一定延续性，可作为本周期的重要参考基线。"
    elif abs(slope_s) > 0 and abs(slope_m) < 1e-9 and abs(slope_l) < 1e-9:
        stability = "仅短周期均线出现明显拐动，中长期仍趋平，可能是局部波动或假突破，需谨慎放大短线信号。"
    else:
        stability = "均线方向不一致，说明多空力量正在博弈，趋势稳定性一般，应结合其他指标与风险控制。"

    text = (
        "📈 移动平均线分析（趋势基线）：\n"
        f"- 使用 {short} / {mid} / {long} 周期简单移动平均线（SMA）衡量短期、中期与长周期趋势。\n"
        f"- 当前均线结构：{structure}\n"
        f"- 价格位置评估：{pos_desc}\n"
        f"- 趋势稳定性判断：{stability}\n"
    )

    return text

def generate_momentum_analysis(price_data):
    """
    从 price_data['technical_data'] 中提取 RSI、MACD、信号线，生成面向 LLM 的动量指标分析文本。
    不进行指标计算，仅做语义解释。

    参数:
        price_data: dict
            包含 'technical_data' 字段的行情数据（参见 get_btc_ohlcv_enhanced 返回结构）
    """
    if not price_data or "technical_data" not in price_data:
        return "📊 动量指标分析：缺少技术指标数据，无法进行动量判断。"

    tech = price_data.get("technical_data", {})
    rsi = tech.get("rsi")
    macd = tech.get("macd")
    signal = tech.get("macd_signal")
    hist = tech.get("macd_histogram")

    # --- 数据可用性检查 ---
    if rsi is None or macd is None or signal is None:
        return "📊 动量指标分析：RSI 或 MACD 数据缺失，暂无法提供有效动量信号。"

    # --- RSI 分析 ---
    if rsi >= 80:
        rsi_desc = "RSI 处于极端超买区，短期上涨透支，存在回调风险。"
    elif rsi >= 70:
        rsi_desc = "RSI 处于超买区，多头动能强，但追高需谨慎。"
    elif 60 <= rsi < 70:
        rsi_desc = "RSI 位于中性偏强区，多头略占优势。"
    elif 40 <= rsi < 60:
        rsi_desc = "RSI 接近中性，多空力量均衡，市场可能处于震荡阶段。"
    elif 30 <= rsi < 40:
        rsi_desc = "RSI 位于中性偏弱区，空头略占上风。"
    elif 20 <= rsi < 30:
        rsi_desc = "RSI 进入超卖区，存在技术性反弹可能。"
    else:
        rsi_desc = "RSI 处于极端超卖区，短期下跌过度，可能出现强势反弹。"

    # --- MACD 分析 ---
    if macd > signal:
        macd_state = "MACD 主线高于信号线，多头动能占优。"
        if hist and hist > 0:
            macd_desc = "多头柱体持续放大，动能延续良好。"
        elif hist and hist < 0:
            macd_desc = "虽然主线高于信号线，但柱体转负，显示上行动能减弱。"
        else:
            macd_desc = "动能维持正向但无明显放大。"
    elif macd < signal:
        macd_state = "MACD 主线低于信号线，空头动能占优。"
        if hist and hist < 0:
            macd_desc = "空头柱体放大，趋势压力明显。"
        elif hist and hist > 0:
            macd_desc = "尽管主线低于信号线，但柱体转正，空头动能出现减弱迹象。"
        else:
            macd_desc = "动能偏空但趋于平缓。"
    else:
        macd_state = "MACD 与信号线几乎重合，动能方向暂不明朗。"
        macd_desc = "市场处于动能转换或整理阶段。"

    # --- 综合结论（LLM友好标签） ---
    if rsi >= 60 and macd > signal:
        overall = "整体动能评估：多头动能占优，市场偏强，可关注延续性。"
    elif rsi <= 40 and macd < signal:
        overall = "整体动能评估：空头动能占优，短期承压，宜谨慎操作。"
    elif 45 <= rsi <= 55:
        overall = "整体动能评估：动能中性，方向不明，适合等待突破信号。"
    else:
        overall = "整体动能评估：多空信号交织，市场处于转换期，宜结合趋势结构观察。"

    text = (
        "📊 动量指标分析：\n"
        f"- RSI：{rsi:.2f}。{rsi_desc}\n"
        f"- MACD 主线：{macd:.4f}，信号线：{signal:.4f}。{macd_state}{macd_desc}\n"
        f"- {overall}\n"
        "- 提示：动量信号仅作为辅助依据，应结合均线结构、价格形态与风险控制共同评估。\n"
    )

    return text

def generate_bollinger_analysis(price_data, lookback: int = 40):
    """
    基于 price_data 中已计算好的布林带数据，生成给 LLM 用的布林带语义分析。

    依赖:
        price_data['technical_data']:
            - bb_upper, bb_lower, bb_position
        price_data['full_data'] (可选，用于带宽压缩/扩张判断):
            - bb_upper, bb_lower, bb_middle

    不重新计算技术指标，只做解释与归纳。
    """

    if not price_data or "technical_data" not in price_data:
        return "🎚️ 布林带分析：缺少布林带相关数据，暂无法评估波动区间与相对位置。"

    tech = price_data["technical_data"]
    bb_pos = tech.get("bb_position")
    bb_upper = tech.get("bb_upper")
    bb_lower = tech.get("bb_lower")
    rsi = tech.get("rsi")

    # 基础可用性检查
    if bb_pos is None or bb_upper is None or bb_lower is None:
        return "🎚️ 布林带分析：布林带数据不完整，暂不将其作为本周期的主要决策依据。"

    try:
        bb_pos = float(bb_pos)
        bb_upper = float(bb_upper)
        bb_lower = float(bb_lower)
    except (TypeError, ValueError):
        return "🎚️ 布林带分析：布林带数据异常，无法给出可靠评估。"

    parts = ["🎚️ 布林带分析："]

    # === 1️⃣ 相对位置解读（使用已给出的 bb_position） ===
    # bb_position = (price - lower) / (upper - lower)
    if bb_pos <= 0.1:
        pos_desc = "价格贴近下轨，处于相对偏弱/可能超卖区域。"
        zone = "下轨附近"
    elif bb_pos <= 0.3:
        pos_desc = "价格位于布林带下半区，偏弱整理或下行趋势中。"
        zone = "下半区"
    elif bb_pos < 0.7:
        pos_desc = "价格接近中轨附近，属于相对均衡/震荡区域。"
        zone = "中部区域"
    elif bb_pos < 0.9:
        pos_desc = "价格位于布林带上半区，表现为偏强运行，多头占优。"
        zone = "上半区"
    else:
        pos_desc = "价格贴近上轨，短期多头情绪较强，可能存在阶段性过热风险。"
        zone = "上轨附近"

    parts.append(f"- 当前位置：约处于区间的 {bb_pos * 100:.2f}%，即{zone}。{pos_desc}")

    # === 2️⃣ 带宽与波动强度（利用 full_data，不做新指标，只对现有列做差） ===
    width_desc = "带宽数据不足，暂不评估波动压缩或扩张。"
    df = price_data.get("full_data")

    try:
        if df is not None and all(col in df.columns for col in ["bb_upper", "bb_lower", "bb_middle"]):
            recent = df.tail(max(lookback, 20)).copy()
            # 避免除零，仅在中轨有效时计算
            recent["bb_width_ratio"] = (recent["bb_upper"] - recent["bb_lower"]) / recent["bb_middle"].replace(0, float("nan"))
            current_row = recent.iloc[-1]
            current_width = float(current_row["bb_width_ratio"]) if pd.notna(current_row["bb_width_ratio"]) else None
            avg_width = float(recent["bb_width_ratio"].dropna().mean()) if not recent["bb_width_ratio"].dropna().empty else None

            if current_width is not None and avg_width is not None:
                if current_width < avg_width * 0.7:
                    width_desc = "当前布林带明显收窄，波动被压缩，后续存在放量突破或单边行情的潜在风险。"
                elif current_width > avg_width * 1.3:
                    width_desc = "当前布林带显著张口，波动放大，多为空头或多头趋势演绎阶段，应重视顺势交易。"
                else:
                    width_desc = "当前布林带带宽接近近期均值，波动水平正常，无明显压缩或极端放大信号。"

    except Exception:
        # 容错，保持默认描述
        pass

    parts.append(f"- 波动带宽评估：{width_desc}")

    # === 3️⃣ 与 RSI 的联合信号（只读已有 RSI，不计算） ===
    overall = None
    try:
        if rsi is not None:
            rsi = float(rsi)
            if bb_pos >= 0.9 and rsi >= 70:
                overall = "综合判断：价格贴近上轨且 RSI 超买，短期存在回调或整理压力，追高需控制仓位与杠杆。"
            elif bb_pos <= 0.1 and rsi <= 30:
                overall = "综合判断：价格贴近下轨且 RSI 超卖，存在技术性反弹或短线修复机会，但需结合趋势确认。"
            elif 0.3 < bb_pos < 0.7 and 40 <= rsi <= 60:
                overall = "综合判断：价格与 RSI 均处于中性区间，更偏向震荡市特征，适合等待突破信号。"

    except (TypeError, ValueError):
        pass

    if not overall:
        overall = "综合判断：布林带当前更多提供价格相对位置与波动信息，应与趋势结构（均线）、MACD、RSI 等联合使用，不单独作为开仓或反手依据。"

    parts.append(f"- {overall}")

    # 风控导向，避免 LLM 把“上轨/下轨”当成机械反转信号
    parts.append("- 提示：价格触及或接近布林带上下轨，并不自动等于反转信号，更重要的是结合成交量、趋势方向和其他指标确认。")

    return "\n".join(parts)

def generate_price_action_tags(price_data: pd.DataFrame) -> list[str]:
    """
    基于本地K线数据生成形态/结构标签。
    仅输出中性标签，不做方向结论（假突破/冲顶等交给大模型判断）。
    """
    if price_data is None or len(price_data) < 20:
        return []

    df = price_data.copy()
    df = df.sort_index()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    tags = set()
    tags.update(_single_candle_tags(df, last, prev))
    tags.update(_sequence_tags(df))
    tags.update(_range_break_tags(df))
    tags.update(_volatility_tags(df))

    return sorted(tags)

def _single_candle_tags(df: pd.DataFrame, last, prev) -> list[str]:
    tags = []

    o, h, l, c = float(last['open']), float(last['high']), float(last['low']), float(last['close'])
    body = abs(c - o)
    full_range = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l

    body_ma_window = min(20, len(df))
    body_ma = (df['close'].iloc[-body_ma_window:] - df['open'].iloc[-body_ma_window:]).abs().mean()

    # 长上下影 & Doji & 大实体
    if upper >= max(2 * body, 0.4 * full_range) and body / full_range <= 0.6:
        tags.append("LONG_UPPER_SHADOW")
    if lower >= max(2 * body, 0.4 * full_range) and body / full_range <= 0.6:
        tags.append("LONG_LOWER_SHADOW")
    if body_ma > 0 and body >= 1.5 * body_ma:
        tags.append("BIG_BODY")
    if body <= 0.2 * full_range and full_range >= 0.5 * body_ma:
        tags.append("SMALL_BODY_DOJI")

    # 吞没候选（仅做线索）
    po, ph, pl, pc = float(prev['open']), float(prev['high']), float(prev['low']), float(prev['close'])
    prev_body = abs(pc - po)

    # 看多吞没候选
    if c > o and pc < po and body > prev_body and l <= pl and c >= ph:
        tags.append("BULLISH_ENGULFING_CANDIDATE")

    # 看空吞没候选
    if c < o and pc > po and body > prev_body and h >= ph and c <= pl:
        tags.append("BEARISH_ENGULFING_CANDIDATE")

    return tags
def _sequence_tags(df: pd.DataFrame) -> list[str]:
    tags = []
    closes = df['close']
    highs = df['high']
    lows = df['low']

    # 连续涨跌（取最近5根内的极值）
    max_lookback = min(5, len(df) - 1)
    up_streak = 0
    down_streak = 0
    for i in range(1, max_lookback + 1):
        if closes.iloc[-i] > closes.iloc[-i-1]:
            up_streak += 1
            if down_streak > 0:
                break
        elif closes.iloc[-i] < closes.iloc[-i-1]:
            down_streak += 1
            if up_streak > 0:
                break
        else:
            break

    if up_streak >= 3:
        tags.append(f"N_CONSECUTIVE_UP_{up_streak}")
    if down_streak >= 3:
        tags.append(f"N_CONSECUTIVE_DOWN_{down_streak}")

    # 高点/低点序列（简单3段结构）
    if len(df) >= 4:
        recent_highs = highs.iloc[-4:]
        recent_lows = lows.iloc[-4:]

        if all(recent_highs.iloc[i] < recent_highs.iloc[i+1] for i in range(3)):
            tags.append("HIGHER_HIGH_SERIES_3")
        if all(recent_lows.iloc[i] > recent_lows.iloc[i+1] for i in range(3)):
            tags.append("LOWER_LOW_SERIES_3")

    # 动能加速：最近5根实体对比前20根
    if len(df) >= 25:
        recent_body = (df['close'].iloc[-5:] - df['open'].iloc[-5:]).abs().mean()
        hist_body = (df['close'].iloc[-25:-5] - df['open'].iloc[-25:-5]).abs().mean()
        if hist_body > 0:
            ratio = recent_body / hist_body
            if ratio >= 1.6:
                # 方向中性，交给模型从趋势+价格判断多空
                tags.append("MOMENTUM_ACCELERATION_STRONG")
            elif ratio >= 1.3:
                tags.append("MOMENTUM_ACCELERATION_MILD")

    return tags
def _range_break_tags(df: pd.DataFrame) -> list[str]:
    tags = []
    closes = df['close']
    highs = df['high']
    lows = df['low']

    last_close = float(closes.iloc[-1])
    last_high = float(highs.iloc[-1])
    last_low = float(lows.iloc[-1])

    # 短&中区间
    short_n = min(48, len(df))
    mid_n = min(144, len(df))

    short_high = float(highs.iloc[-short_n:].max())
    short_low = float(lows.iloc[-short_n:].min())
    mid_high = float(highs.iloc[-mid_n:].max())
    mid_low = float(lows.iloc[-mid_n:].min())

    # 相对距离（永续合约，这里用百分比）
    def rel(x, y):
        return abs(x - y) / max(y, 1e-9)

    # 贴近区间边缘
    if rel(last_close, short_high) <= 0.003:
        tags.append("NEAR_SHORT_RANGE_HIGH")
    if rel(last_close, short_low) <= 0.003:
        tags.append("NEAR_SHORT_RANGE_LOW")

    # 短区间突破
    if last_close > short_high * 1.001:
        tags.append("BREAK_ABOVE_SHORT_RANGE_HIGH")
    if last_close < short_low * 0.999:
        tags.append("BREAK_BELOW_SHORT_RANGE_LOW")

    # 假突破嫌疑特征（仍是“嫌疑”，不是结论）
    # 上破后长上影/收回区间附近
    if "BREAK_ABOVE_SHORT_RANGE_HIGH" in tags:
        upper_shadow = last_high - max(float(df['open'].iloc[-1]), last_close)
        body = abs(last_close - float(df['open'].iloc[-1]))
        full_range = max(last_high - last_low, 1e-9)

        if upper_shadow >= max(2 * body, 0.4 * full_range) or last_close <= short_high * 1.0015:
            tags.append("BREAKUP_WEAK_FOLLOWTHROUGH_HINT")

    if "BREAK_BELOW_SHORT_RANGE_LOW" in tags:
        lower_shadow = min(float(df['open'].iloc[-1]), last_close) - last_low
        body = abs(last_close - float(df['open'].iloc[-1]))
        full_range = max(last_high - last_low, 1e-9)

        if lower_shadow >= max(2 * body, 0.4 * full_range) or last_close >= short_low * 0.9985:
            tags.append("BREAKDOWN_WEAK_FOLLOWTHROUGH_HINT")

    return tags
def _volatility_tags(df: pd.DataFrame) -> list[str]:
    tags = []
    if len(df) < 40:
        return tags

    hl = df['high'] - df['low']

    recent_n = 20
    base_n = 60

    recent_vol = hl.iloc[-recent_n:].mean()
    base_vol = hl.iloc[-base_n:-recent_n].mean() if len(df) >= base_n + recent_n else hl.iloc[:-recent_n].mean()

    if base_vol <= 0:
        return tags

    ratio = recent_vol / base_vol

    if ratio <= 0.6:
        tags.append("VOLATILITY_SQUEEZE")
    elif ratio >= 1.6:
        tags.append("VOLATILITY_EXPANSION")

    return tags
def format_price_action_tags_for_llm(tags: list[str]) -> str:
    """
    将本地形态/结构标签转换为 LLM 友好的简要文字描述。
    要求：
    - 简短
    - 中性
    - 不下交易结论，只描述结构线索
    """
    if not tags:
        return "未检测到特别突出的K线形态或价格结构信号，本地特征提取保持中性。"

    desc_map = {
        # 单根K线
        "LONG_UPPER_SHADOW": "当前K线出现相对明显的长上影，上方抛压或获利了结迹象增加。",
        "LONG_LOWER_SHADOW": "当前K线出现相对明显的长下影，下方承接或买盘支撑迹象增加。",
        "BIG_BODY": "当前K线实体显著大于近期平均，短线方向性波动放大。",
        "SMALL_BODY_DOJI": "当前K线实体较小，短线方向犹豫，等待进一步选择。",
        "BULLISH_ENGULFING_CANDIDATE": "出现潜在多头吞没形态候选，短线多头尝试主导节奏。",
        "BEARISH_ENGULFING_CANDIDATE": "出现潜在空头吞没形态候选，短线空头尝试主导节奏。",

        # 连续结构 / 动能
        "MOMENTUM_ACCELERATION_STRONG": "近期K线实体整体明显放大，相比过去存在较强动能加速迹象。",
        "MOMENTUM_ACCELERATION_MILD": "近期K线实体略有放大，存在一定动能增强迹象。",

        # 区间/突破
        "NEAR_SHORT_RANGE_HIGH": "当前价格逼近近期短周期震荡区间上沿位置。",
        "NEAR_SHORT_RANGE_LOW": "当前价格逼近近期短周期震荡区间下沿位置。",
        "BREAK_ABOVE_SHORT_RANGE_HIGH": "价格向上突破近期短周期区间上沿，有上攻延伸的尝试。",
        "BREAK_BELOW_SHORT_RANGE_LOW": "价格向下跌破近期短周期区间下沿，有下探延伸的尝试。",
        "BREAKUP_WEAK_FOLLOWTHROUGH_HINT": "上破后跟随力度相对有限，存在动能衰减或假突破的结构疑虑。",
        "BREAKDOWN_WEAK_FOLLOWTHROUGH_HINT": "下破后跟随力度相对有限，存在动能衰减或假跌破的结构疑虑。",

        # 波动结构
        "VOLATILITY_SQUEEZE": "近期波动率明显收缩，市场处于压缩整理阶段，潜在蓄势状态。",
        "VOLATILITY_EXPANSION": "近期波动率明显放大，市场处于活跃波动阶段，方向博弈加剧。",
    }

    # 支持 N_CONSECUTIVE_UP_x / DOWN_x 动态文案
    pretty_lines = []

    for t in tags:
        if t.startswith("N_CONSECUTIVE_UP_"):
            n = t.split("_")[-1]
            pretty_lines.append(f"近期出现连续 {n} 根收盘抬升的上涨序列，多头短线保持主动。")
        elif t.startswith("N_CONSECUTIVE_DOWN_"):
            n = t.split("_")[-1]
            pretty_lines.append(f"近期出现连续 {n} 根收盘走低的下跌序列，空头短线保持主动。")
        elif t in desc_map:
            pretty_lines.append(desc_map[t])
        # 未映射的标签静默忽略或保留原名（建议忽略，避免噪音）

    if not pretty_lines:
        return "存在部分结构标签触发，但整体信号不具备单独解释意义，请综合其他因子评估。"

    return "\n".join(f"- {line}" for line in pretty_lines)


def evaluate_overheat(price_data):
    """
    基于已有技术数据，给出一个“动能是否可能透支”的评估结果。
    仅作为特征输入给大模型，不是硬风控规则。

    返回:
        {
            "level": "none" | "mild" | "strong",
            "factors": [str, ...]  # 描述原因，供拼接进 prompt
        }
    """
    tech = price_data.get("technical_data", {}) or {}
    rsi = tech.get("rsi")
    bb_pos = tech.get("bb_position")
    macd_hist = tech.get("macd_histogram")
    sma_5 = tech.get("sma_5")
    sma_20 = tech.get("sma_20")

    factors = []

    try:
        if rsi is not None:
            rsi = float(rsi)
        if bb_pos is not None:
            bb_pos = float(bb_pos)
        if macd_hist is not None:
            macd_hist = float(macd_hist)
        if sma_5 is not None and sma_20 is not None:
            sma_5 = float(sma_5)
            sma_20 = float(sma_20)
    except (TypeError, ValueError):
        return {"level": "none", "factors": ["技术数据异常，未进行透支评估"]}

    # 1) 价格相对布林带的位置
    if bb_pos is not None:
        if bb_pos >= 1.05:
            factors.append("价格明显高于布林上轨")
        elif bb_pos >= 0.95:
            factors.append("价格接近布林带上沿")

    # 2) RSI 高位区
    if rsi is not None:
        if rsi >= 80:
            factors.append("RSI 处于极高水平")
        elif rsi >= 70:
            factors.append("RSI 处于高位区间")

    # 3) 均线加速或乖离（简单看 5 与 20 的差）
    if sma_5 and sma_20:
        diff_ratio = (sma_5 - sma_20) / sma_20 if sma_20 != 0 else 0
        if diff_ratio > 0.03:
            factors.append("短期价格/均线相对中期均线乖离偏大")

    # 4) MACD 柱体衰减（需要 full_data，看最近几根）
    df = price_data.get("full_data")
    if df is not None and "macd_histogram" in df.columns:
        recent = df["macd_histogram"].tail(4).tolist()
        if len([x for x in recent if x is not None]) >= 3:
            # 简单判断：从正高值开始走低，或在高位缩短
            cleaned = [float(x) for x in recent if x is not None]
            if len(cleaned) >= 3 and cleaned[-1] < cleaned[-2] > cleaned[-3] and cleaned[-2] > 0:
                factors.append("MACD 动能在高位出现减弱迹象")

    # 归纳 level（温和，不当成铁律，只是语义标签）
    strong_signals = [
        "价格明显高于布林上轨",
        "RSI 处于极高水平",
        "MACD 动能在高位出现减弱迹象",
        "短期价格/均线相对中期均线乖离偏大",
    ]

    if not factors:
        level = "none"
    else:
        score = sum(1 for f in factors if f in strong_signals)
        if score >= 3:
            level = "strong"
        elif score >= 1:
            level = "mild"
        else:
            level = "none"

    return {"level": level, "factors": factors}

def evaluate_price_volume_pattern(price_data, lookback: int = 20):
    """
    基于最近K线的价格与成交量关系，评估当前是否更像：
    - 有支撑的有效突破（clean_breakout）
    - 可能的假突破/冲高回落（possible_fake_breakout）
    - 动能不足的弱突破（weak_breakout）
    - 普通震荡/无明显信号（normal）

    仅用于给大模型提供结构化线索，不直接做交易决策。
    """
    df = price_data.get("full_data")
    if df is None:
        return {"label": "normal", "reasons": ["缺少完整K线数据，未评估量价形态"]}

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        return {"label": "normal", "reasons": ["K线数据缺少必要字段，未评估量价形态"]}

    if len(df) < lookback + 3:
        return {"label": "normal", "reasons": ["历史样本不足，量价评估不具稳定性"]}

    recent = df.tail(lookback + 2).copy()
    last = recent.iloc[-1]
    prev = recent.iloc[-2]
    hist = recent.iloc[:-1]

    try:
        o, h, l, c, v = map(float, (last["open"], last["high"], last["low"], last["close"], last["volume"]))
        prev_high_max = float(hist["high"].max())
        avg_vol = float(hist["volume"].mean())
    except Exception:
        return {"label": "normal", "reasons": ["量价数据异常，未评估量价形态"]}

    if avg_vol <= 0:
        return {"label": "normal", "reasons": ["平均成交量异常，未评估量价形态"]}

    # 基本形态特征
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    vol_ratio = v / avg_vol

    # 是否创新高（略加缓冲避免噪点）
    is_new_high = h > prev_high_max * 1.001

    reasons = []

    # 情况 1：有效突破（新高 + 强收盘 + 放量）
    if is_new_high and c > (l + 0.75 * rng) and vol_ratio >= 1.2:
        reasons.append("价格突破近期高点且收盘接近高位，成交量高于均值，突破相对有支撑。")
        return {"label": "clean_breakout", "reasons": reasons}

    # 情况 2：可能假突破（新高但收回、长上影、高位放量）
    if is_new_high:
        # 长上影 + 放量
        if upper_shadow > max(body * 2, rng * 0.4) and vol_ratio >= 1.0:
            reasons.append("出现高位长上影放量冲高回落，存在假突破或短线资金出货可能。")
            return {"label": "possible_fake_breakout", "reasons": reasons}

        # 新高但缩量
        if vol_ratio < 0.8:
            reasons.append("价格略创新高但成交量不足，突破动能偏弱。")
            return {"label": "weak_breakout", "reasons": reasons}

    # 情况 3：无明显突破，但有信息
    if vol_ratio >= 1.5 and body < rng * 0.3 and upper_shadow > body and c < (l + 0.5 * rng):
        reasons.append("放量但收盘偏弱，存在上方压力或分歧。")
        return {"label": "possible_fake_breakout", "reasons": reasons}

    if vol_ratio <= 0.7 and body < rng * 0.3:
        reasons.append("缩量小实体K线，市场观望情绪较重。")

    if not reasons:
        reasons.append("量价关系未出现明显异常或突破信号，视为常规波动。")

    return {"label": "normal", "reasons": reasons}

def compute_risk_reward_for_sides(price_data,
                                  lookback: int = 80,
                                  recent_exclude: int = 8,
                                  breakout_eps: float = 0.001) -> dict:
    """
    基于最近一段结构，分别评估做多与做空方向的区间型风险回报。
    显式区分：
    - 区间内交易（range mode）
    - 向上/向下突破后的交易（breakout mode）

    返回:
    {
        "mode": "range" | "up_breakout" | "down_breakout",
        "long":  {...},
        "short": {...},
    }
    参数 price_data 可直接传入包含 OHLCV 列的 DataFrame，或是包含 'full_data' 键的行情字典。
    """

    if price_data is None:
        return {
            "mode": "range",
            "long":  {"tag": "unknown", "ratio": None, "reason": "缺少K线数据，无法评估风险回报结构"},
            "short": {"tag": "unknown", "ratio": None, "reason": "缺少K线数据，无法评估风险回报结构"},
        }

    if isinstance(price_data, pd.DataFrame):
        df = price_data.copy()
    else:
        df = price_data.get("full_data") if isinstance(price_data, dict) else None
        if df is not None:
            df = df.copy()

    if df is None or len(df) < (lookback + recent_exclude + 5):
        return {
            "mode": "range",
            "long":  {"tag": "unknown", "ratio": None, "reason": "样本不足，无法稳定评估风险回报结构"},
            "short": {"tag": "unknown", "ratio": None, "reason": "样本不足，无法稳定评估风险回报结构"},
        }

    df = df.iloc[-(lookback + recent_exclude):]  # 保留需要的窗口
    recent = df.iloc[-recent_exclude:]
    base = df.iloc[:-recent_exclude]             # 用于定义“原始区间”

    prev_high = float(base["high"].max())
    prev_low = float(base["low"].min())
    current = float(recent["close"].iloc[-1])

    base_range = max(prev_high - prev_low, 1e-8)

    # --- 检测突破状态 ---
    up_break = current > prev_high * (1 + breakout_eps)
    down_break = current < prev_low * (1 - breakout_eps)

    def _tag(r: float) -> str:
        if r >= 2.0:
            return "favorable"
        elif r >= 1.0:
            return "neutral"
        elif r > 0:
            return "unfavorable"
        else:
            return "unknown"

    # === 情况1：向上突破（up_breakout mode） ===
    if up_break:
        # 假设：上破有效，多头止损放在 prev_high 下方，目标以“原区间高度的测幅”估计
        breakout_level = prev_high
        projected_target = breakout_level + base_range  # 机械测幅，非预测，只给结构参考

        risk_long = max(current - breakout_level, 1e-8)
        reward_long = max(projected_target - current, 0.0)
        ratio_long = reward_long / risk_long if reward_long > 0 else 0.0

        # 逆势做空：视为结构上不利或高度不确定
        # 不给它“看起来很香”的R:R，避免误导
        risk_short = max(projected_target - current, 1e-8)
        reward_short = max(current - breakout_level, 0.0)
        ratio_short = reward_short / risk_short if reward_short > 0 else 0.0

        return {
            "mode": "up_breakout",
            "long": {
                "tag": _tag(ratio_long),
                "ratio": round(ratio_long, 2),
                "reason": (
                    f"价格已明显上破前高区间（{prev_low:.1f}~{prev_high:.1f}），"
                    f"多头参考以前高作为止损附近位置，以原区间高度做测幅，"
                    f"当前上破后的结构性R:R约为 {ratio_long:.2f}。"
                ),
            },
            "short": {
                # 这里直接把大部分情况压成不利/未知
                "tag": "unfavorable" if ratio_short < 1.0 else "neutral",
                "ratio": round(ratio_short, 2),
                "reason": (
                    "当前处于上破区间后的高位，逆势做空属于反趋势博弈，"
                    "即使短线R:R看似可观，也不应视为结构性优势，仅在多因子强烈反转信号下谨慎考虑。"
                ),
            },
        }

    # === 情况2：向下突破（down_breakout mode） ===
    if down_break:
        breakout_level = prev_low
        projected_target = breakout_level - base_range

        risk_short = max(breakout_level - current, 1e-8)
        reward_short = max(current - projected_target, 0.0)
        ratio_short = reward_short / risk_short if reward_short > 0 else 0.0

        risk_long = max(current - projected_target, 1e-8)
        reward_long = max(breakout_level - current, 0.0)
        ratio_long = reward_long / risk_long if reward_long > 0 else 0.0

        return {
            "mode": "down_breakout",
            "long": {
                "tag": "unfavorable" if ratio_long < 1.0 else "neutral",
                "ratio": round(ratio_long, 2),
                "reason": (
                    "当前处于下破区间后的低位，逆势做多属于反趋势博弈，"
                    "结构上并不具备稳定优势，仅在出现明显止跌与多因子共振时才可谨慎评估。"
                ),
            },
            "short": {
                "tag": _tag(ratio_short),
                "ratio": round(ratio_short, 2),
                "reason": (
                    f"价格已明显跌破前低区间（{prev_low:.1f}~{prev_high:.1f}），"
                    f"空头参考以前低作为止损上方区域，以原区间高度做测幅，"
                    f"当前下破后的结构性R:R约为 {ratio_short:.2f}。"
                ),
            },
        }

    # === 情况3：未突破，正常区间内（range mode） ===
    # 回到对称结构
    current_range_high = float(df["high"].max())
    current_range_low = float(df["low"].min())
    current_range_span = max(current_range_high - current_range_low, 1e-8)

    risk_long = max(current - current_range_low, 1e-8)
    reward_long = max(current_range_high - current, 0.0)
    ratio_long = reward_long / risk_long if reward_long > 0 else 0.0

    risk_short = max(current_range_high - current, 1e-8)
    reward_short = max(current - current_range_low, 0.0)
    ratio_short = reward_short / risk_short if reward_short > 0 else 0.0

    long_pct_risk = risk_long / current_range_span
    long_pct_reward = reward_long / current_range_span
    short_pct_risk = risk_short / current_range_span
    short_pct_reward = reward_short / current_range_span

    return {
        "mode": "range",
        "long": {
            "tag": _tag(ratio_long),
            "ratio": round(ratio_long, 2),
            "reason": (
                f"当前价格位于近期区间内，做多参考区间低点作为风险边界，"
                f"下方风险约占区间 {long_pct_risk:.1%}，上方空间约占 {long_pct_reward:.1%}。"
            ),
        },
        "short": {
            "tag": _tag(ratio_short),
            "ratio": round(ratio_short, 2),
            "reason": (
                f"当前价格位于近期区间内，做空参考区间高点作为风险边界，"
                f"上方风险约占区间 {short_pct_risk:.1%}，下方空间约占 {short_pct_reward:.1%}。"
            ),
        },
    }

def _translate_rr_tag(tag: str) -> str:
    mapping = {
        "favorable": "相对有利",
        "neutral": "中性",
        "unfavorable": "相对不利",
        "unknown": "信息不足",
    }
    return mapping.get(tag, "中性")

def format_risk_reward_for_prompt(rr: dict, trend_summary: str | None = None) -> str:
    """
    将双向R:R结果转为给 LLM 的自然语言说明。
    trend_summary 可选：可传入你已有的趋势描述，提示模型“优先参考顺势一侧”。
    """
    long_info = rr.get("long", {})
    short_info = rr.get("short", {})

    long_tag = _translate_rr_tag(long_info.get("tag"))
    short_tag = _translate_rr_tag(short_info.get("tag"))

    long_ratio = long_info.get("ratio")
    short_ratio = short_info.get("ratio")

    # 字符串兜底，避免 None 拼接出错
    long_ratio_str = f"{long_ratio:.2f}" if isinstance(long_ratio, (int, float)) else "?"
    short_ratio_str = f"{short_ratio:.2f}" if isinstance(short_ratio, (int, float)) else "?"

    lines = [
        "【风险回报结构】（多空分向评估，仅基于区间结构，不代表必然走势）"]
    

    mode = rr.get("mode", "range")

    if mode == "up_breakout":
        lines.append("- 当前处于上破区间后的延伸阶段，请优先从多头角度评估结构是否健康，逆势做空仅在强烈反转信号下考虑。")
    elif mode == "down_breakout":
        lines.append("- 当前处于下破区间后的延伸阶段，请优先从空头角度评估结构是否健康，逆势做多仅在强烈止跌信号下考虑。")
    else:
        lines.append("- 当前价格尚在近期震荡区间内，可对多空方向分别从区间上下沿角度评估R:R。")

    # 然后附上 long/short 的 tag、ratio、reason（保持我们上版风格）

    lines += [
        f"- 做多方向: {long_tag}（理论R:R≈{long_ratio_str}），{long_info.get('reason', '')}",
        f"- 做空方向: {short_tag}（理论R:R≈{short_ratio_str}），{short_info.get('reason', '')}",
        "",
        "使用指引：",
        "1. 优先结合当前趋势方向，参考与趋势同向一侧的风险回报；",
        "2. 若某一方向为“相对不利”，仅在多因子强烈共振时才考虑；",
        "3. 该评估不包含你的主观预测，仅提供区间结构上的风险/空间对比。",
    ]

    if trend_summary:
        lines.append(f"4. 当前趋势概览：{trend_summary.strip()}")

    return "\n".join(lines)

def calculate_intelligent_position(signal_data, price_data, current_position):
    """计算智能仓位大小 - 修复版"""
    config = TRADE_CONFIG['position_management']

    # 🆕 新增：如果禁用智能仓位，使用固定仓位
    if not config.get('enable_intelligent_position', True):
        fixed_contracts = 0.1  # 固定仓位大小，可以根据需要调整
        print(f"🔧 智能仓位已禁用，使用固定仓位: {fixed_contracts} 张")
        return fixed_contracts

    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']

        # 基于账户资金与最小下单约束，动态计算下单基数（确保可开单）
        dynamic_base = compute_dynamic_base_usdt(
            exchange,
            TRADE_CONFIG['symbol'],
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG.get('contract_size', 0.01),
            TRADE_CONFIG.get('min_amount', 0.01),
            config['base_usdt_amount']
        )
        base_usdt = dynamic_base or config['base_usdt_amount']
        print(f"💰 可用USDT余额: {usdt_balance:.2f}, 下单基数(动态): {base_usdt:.2f}")

        # 根据信心程度调整 - 修复这里
        confidence_multiplier = {
            'HIGH': config['high_confidence_multiplier'],
            'MEDIUM': config['medium_confidence_multiplier'],
            'LOW': config['low_confidence_multiplier']
        }.get(signal_data['confidence'], 1.0)  # 添加默认值

        # 根据趋势强度调整
        trend = price_data['trend_analysis'].get('overall', '震荡整理')
        if trend in ['强势上涨', '强势下跌']:
            trend_multiplier = config['trend_strength_multiplier']
        else:
            trend_multiplier = 1.0

        # 根据RSI状态调整（超买超卖区域减仓）
        rsi = price_data['technical_data'].get('rsi', 50)
        if rsi > 75 or rsi < 25:
            rsi_multiplier = 0.7
        else:
            rsi_multiplier = 1.0

        # 计算建议投入USDT金额
        suggested_usdt = base_usdt * confidence_multiplier * trend_multiplier * rsi_multiplier

        # 风险管理：不超过总资金的指定比例 - 删除重复定义
        max_usdt = usdt_balance * config['max_position_ratio']
        final_usdt = min(suggested_usdt, max_usdt)

        # 正确的合约张数计算！
        # 公式：合约张数 = (投入USDT) / (当前价格 * 合约乘数)
        contract_size = (final_usdt) / (price_data['price'] * TRADE_CONFIG['contract_size'])

        print(f"📊 仓位计算详情:")
        print(f"   - 基础USDT: {base_usdt}")
        print(f"   - 信心倍数: {confidence_multiplier}")
        print(f"   - 趋势倍数: {trend_multiplier}")
        print(f"   - RSI倍数: {rsi_multiplier}")
        print(f"   - 建议USDT: {suggested_usdt:.2f}")
        print(f"   - 最终USDT: {final_usdt:.2f}")
        print(f"   - 合约乘数: {TRADE_CONFIG['contract_size']}")
        print(f"   - 计算合约: {contract_size:.4f} 张")

        # 精度处理：OKX BTC合约最小交易单位为0.01张
        contract_size = round(contract_size, 2)  # 保留2位小数

        # 确保最小交易量
        min_contracts = TRADE_CONFIG.get('min_amount', 0.01)
        if contract_size < min_contracts:
            contract_size = min_contracts
            print(f"⚠️ 仓位小于最小值，调整为: {contract_size} 张")

        print(f"🎯 最终仓位: {final_usdt:.2f} USDT → {contract_size:.2f} 张合约")
        return contract_size

    except Exception as e:
        print(f"❌ 仓位计算失败，使用基础仓位: {e}")
        # 紧急备用计算
        base_usdt = config['base_usdt_amount']
        contract_size = (base_usdt * TRADE_CONFIG['leverage']) / (
                    price_data['price'] * TRADE_CONFIG.get('contract_size', 0.01))
        return round(max(contract_size, TRADE_CONFIG.get('min_amount', 0.01)), 2)


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_80'] = df['close'].rolling(window=80, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}


def get_sentiment_indicators():
    """获取情绪指标 - 简洁版本"""
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "7ad48a56-8730-4238-a714-eebc30834e3e"

        # 获取最近4小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # 只保留核心指标
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": get_sentiment_tokens(),
        }

        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = requests.post(API_URL, json=request_body, headers=headers, timeout=5)
        if response.status_code != 200:
            print(f"⚠️ 情绪API状态码异常: {response.status_code}")
            return None

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                # 查找第一个有有效数据的时间段
                for period in time_periods:
                    period_data = period.get("data", [])

                    sentiment = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = item.get("value", "").strip()

                        if value:  # 只处理非空值
                            try:
                                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                    sentiment[endpoint] = float(value)
                                    valid_data_found = True
                            except (ValueError, TypeError):
                                continue

                    # 如果找到有效数据
                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment['CO-A-02-01']
                        negative = sentiment['CO-A-02-02']
                        net_sentiment = positive - negative

                        # 正确的时间延迟计算
                        data_delay = int((datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                        print(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")

                        return {
                            'positive_ratio': positive,
                            'negative_ratio': negative,
                            'net_sentiment': net_sentiment,
                            'data_time': period['startTime'],
                            'data_delay_minutes': data_delay
                        }

                print("❌ 所有时间段数据都为空")
                return None

        return None
    except Exception as e:
        print(f"情绪指标获取失败: {e}")
        return None


def get_sentiment_indicators_with_retry(max_retries: int = 2, delay_sec: int = 1):
    """对情绪API做轻量重试，失败则降级为None。"""
    for attempt in range(max_retries):
        data = get_sentiment_indicators()
        if data:
            return data
        time.sleep(delay_sec)
    print("⚠️ 情绪指标暂不可用，已降级为技术分析-only")
    return None


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_80'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}


def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df = calculate_technical_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(TRADE_CONFIG['recent_kline_count']).to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_80': current_data.get('sma_80', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"获取增强K线数据失败: {e}")
        return None


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})
    sma_analysis_text = generate_sma_analysis(price_data)
    momentum_analysis_text = generate_momentum_analysis(price_data)
    boll_text = generate_bollinger_analysis(price_data)
    overheat = evaluate_overheat(price_data)
    pvp = evaluate_price_volume_pattern(price_data)
    risk_reward = compute_risk_reward_for_sides(price_data)
    risk_reward_text = format_risk_reward_for_prompt(risk_reward, trend_summary=None)

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value and pd.notna(value) else default

    analysis_text = f"""
    【技术指标分析】
    {sma_analysis_text}

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向（提供趋势动能强度判断）: {trend.get('macd', 'N/A')}

    {momentum_analysis_text}

    {boll_text}

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}

    【动能透支评估 - 系统辅助信息】
        - 当前透支等级: {overheat["level"]}
        - 参考信号: { "；".join(overheat["factors"]) if overheat["factors"] else "无明显透支信号" }
    
    【量价结构评估】
        - 当前形态标签: {pvp['label']}
        - 参考说明: {"；".join(pvp["reasons"]) if pvp.get("reasons") else "无明显异常信号"}

    {risk_reward_text}
    """
    return analysis_text


def get_current_position():
    """获取当前持仓情况 - OKX版本"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原始内容: {json_str}")
            print(f"错误详情: {e}")
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data['price'] * 0.98,  # -2%
        "take_profit": price_data['price'] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }

def format_sentiment_text(sentiment_data):
        if not sentiment_data:
            return "【市场情绪】数据暂不可用"

        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        base = (
            f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} "
            f"悲观{sentiment_data['negative_ratio']:.1%} "
            f"净值{sign}{sentiment_data['net_sentiment']:.3f}"
        )

        delay = sentiment_data.get("data_delay_minutes", None)
        if delay is None:
            # 没有延迟信息就不多说
            return base

        # 新鲜度分级（本地机械逻辑）
        if delay <= 15:
            freshness = "（情绪数据较新，可作为辅助验证信号使用。）"
        elif delay <= 45:
            freshness = "（情绪数据存在一定延迟，仅作参考，不得单独作为交易依据。）"
        elif delay <= 90:
            freshness = "（情绪数据明显滞后，仅作背景信息，不应提升做多或做空信心。）"
        else:
            freshness = "（情绪数据严重滞后，本次决策请忽略情绪信号，专注技术面。）"

        return base + " " + freshness

def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本
    # recent_n = TRADE_CONFIG.get('recent_kline_count', 20)
    # kline_text = f"【最近{recent_n}根{TRADE_CONFIG['timeframe']}K线数据(K线{recent_n}为最新数据)】\n"
    # for i, kline in enumerate(price_data['kline_data'][-recent_n:]):
    #     trend = "阳线" if kline['close'] > kline['open'] else "阴线"
    #     change = ((kline['close'] - kline['open']) / kline['open']) * 100
    #     kline_text += f"    K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"
    price_action_tags = generate_price_action_tags(price_data)
    price_action_text = "   【K线形态或价格结构信号】\n     " + format_price_action_tags_for_llm(price_action_tags)

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n  【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 获取情绪数据
    sentiment_data = get_sentiment_indicators_with_retry()
    sentiment_text = format_sentiment_text(sentiment_data)

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
    pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']:.2f} USDT" if current_pos else ""

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下{get_human_pair()} {TRADE_CONFIG['timeframe']}周期数据进行分析：

    【趋势优先决策矩阵】
        当以下条件同时满足：
            - 多个周期趋势同向（例如短期与中期均线均为空头或多头）；
            - MACD 与趋势方向一致；
            - 动能透支等级为 none；
            - 量价结构为 normal 或 clean；
        则本次决策应倾向顺势方向给出 BUY 或 SELL 信号，且置信度至少为 MEDIUM，除非系统已明确标记风险回报结构明确不利。

    【市场分析通用原则】
        - 趋势信息是主导信号，但应结合动量与结构确认，避免过度解读单一强势。
        - 若多项指标方向一致，可提高趋势信心；若指标冲突，应保持中性判断或观望。
        - 动能、布林带、成交量等信号用于验证趋势的可持续性，而非替代趋势本身。
        - 请基于技术面、结构与系统提供的辅助信息（透支、情绪、量价），综合评估信号强度。

    【风险与仓位建议】
        - 根据系统输出的综合信号，合理判断信心等级（HIGH / MEDIUM / LOW）。
        - 当检测到透支、假突破或情绪滞后时，应降低仓位或选择 HOLD。
        - 当信号一致且风险低时，可在对应方向中等仓位参与，但仍应提供风险说明。
        - 不需要引用历史持仓、盈亏或账户信息，它们不在模型输入范围内。

    【结构化辅助说明】
        系统还会提供以下内容供参考：
            - 动能透支等级（none/mild/strong）
            - 情绪信号新鲜度说明
            - 量价结构评估标签（clean/weak/fake/normal）
            - 你应将这些内容作为上下文的结构化提示信息使用，而非机械指令。请始终以解释性思维阐述交易理由。

    【趋势与风险平衡原则】
        1. 趋势优先，但要识别“透支风险”：
            - 当短期与中期均线方向一致、价格沿同一方向运行时，可以优先考虑顺势交易（无论多空）。
            - 但如果此时多项信号同时指向“行情可能已经接近阶段尾声”（例如：价格连续创高/创低但动能放缓、动量指标处于极值区间、价格多次触及通道边缘等），你需要降低顺势方向的信心，而不是简单视为更强信号。
        2. 请主动识别以下“可能透支”的组合特征（不限于固定阈值）：
            - 价格处于近期波动区间的极端位置（上沿或下沿）；
            - 动量指标在极值区域但边际增量减弱（如MACD柱体缩短、RSI在高位或低位横盘等）；
            - 突破后缺乏持续跟进（如放量冲高后回落、放量杀跌后拉回、影线明显等）。
            遇到这些情况，你应更偏向：
                - 降低信号置信度；
                - 建议小仓或观望；
                - 给出“等待更好入场位置”的理由。
        3. 在趋势延续且无明显透支迹象时：
            - 你可以给出与趋势同向的BUY或SELL信号，并根据技术结构和波动环境给出合理的置信度和仓位建议。
            - 不需要机械依赖某一个指标的单点阈值，而是综合评估多项信息的一致性与可持续性。
        4. 若技术信号之间存在明显冲突：
            - 例如：趋势看多，但多项信号提示可能见顶或动能衰减，
            - 优先选择更保守的方案（降低置信度、小仓或HOLD），并在理由中说明冲突点。

    【K线形态与结构线索使用原则】
        将这些标签视为：
            - 判断突破有效性/假突破嫌疑
            - 判断冲顶/衰竭/趋势延续/震荡犹豫
        的辅助证据，而不是机械信号。
        如认为存在明显假突破或冲顶迹象，请在推理说明中指出“对应的标签依据”，并在最终JSON决策中体现你的判断。

    【动能透支处理原则】
        你会收到一段“动能透支评估 - 系统辅助信息”，其中包含 level（none/mild/strong）以及参考信号说明。
        请按以下方式理解和使用（这是思考方向，而不是死规则）：

        - 若 level = "strong":
        - 优先考虑这是阶段性高风险区域；
        - 降低顺势信心，倾向小仓或观望，而不是给出 HIGH 信心的单边信号；
        - 如仍认为可以顺势操作，必须在理由中清晰说明为何当前结构仍支持继续跟进。

        - 若 level = "mild":
        - 说明部分极值或动能放缓迹象，需要更谨慎评估；
        - 可以给出顺势信号（BUY/SELL），但不应简单视为“无脑强势”，应考虑更合理的仓位与保护。

        - 若 level = "none":
        - 说明当前不存在明显透支信号，可以更专注于趋势与结构本身的判断，无论是看多还是看空。

        在任何情况下，请综合 K线结构、趋势、动量、布林带和情绪，不要因为“强势”或“单一信号”就给出激进决策。
    
    【情绪信号使用原则】
        - 你会在【市场情绪】后面看到一段关于“数据是否新鲜”的说明（例如：数据较新 / 存在延迟 / 明显滞后 / 请忽略情绪）。
        - 当说明为“数据较新”时，可以将情绪视为技术信号的辅助放大因素，前提是技术面本身合理。
        - 当说明为“存在延迟”或“明显滞后”时，情绪只能作为背景信息，不得单独提高做多或做空的置信度。
        - 当说明为“请忽略情绪信号”时，你在本次决策中应完全基于技术面与结构，不使用情绪作为加分项。
        - 不需要根据具体分钟数做机械判断，请根据说明语义综合考量。

    【量价与突破信号使用原则】
        - 你会看到一段【量价结构评估】，其中包含:
            - 当前形态标签(label): clean_breakout / possible_fake_breakout / weak_breakout / normal
            - 若干参考说明(reasons)。
        - 当标签为 clean_breakout 时：
            - 可以更信任当前突破的有效性，但仍需结合趋势与风险管理，不等于盲目追涨或杀跌。
        - 当标签为 possible_fake_breakout 或 weak_breakout 时：
            - 请优先考虑这是一个需要谨慎对待的位置：
                - 倾向降低顺势信心、控制仓位，或选择观望；
                - 如仍选择顺势参与，须在理由中清晰说明为何认为是假信号或风险可控。
        - 当标签为 normal 时：
            - 说明当前量价关系中性，你可以主要依据趋势、动量和结构来决策。
        以上内容是供你参考的结构化线索，而不是机械规则。请在综合全部上下文后，给出有解释的交易判断。
    
    【重要】请基于技术分析做出明确判断，避免因过度谨慎而错过趋势行情！

    【分析要求】
    基于以上规则，结合后续我提供的实盘数据，请给出明确的交易信号

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简要分析理由(包含趋势判断和技术依据)",
        "stop_loss": 具体价格,
        "take_profit": 具体价格, 
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    
    ---------------以下是实盘数据部分-------------------------

    {price_action_text}

    {technical_analysis}

    {signal_text}

    {sentiment_text}  # 添加情绪分析

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} {get_contract_unit_name()}
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}{pnl_text}

    """

    # 可选打印构造的Prompt，便于调试与复查
    if TRADE_CONFIG.get('print_prompt'):
        try:
            print("\n===== DeepSeek Prompt Begin =====")
            print(prompt)
            print("===== DeepSeek Prompt End =====\n")
        except Exception as e:
            print(f"⚠️ 打印Prompt失败: {e}")

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位专业的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        print(f"DeepSeek原始回复: {result}")

        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)

            if signal_data is None:
                signal_data = create_fallback_signal(price_data)
        else:
            signal_data = create_fallback_signal(price_data)

        # 验证必需字段
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            signal_data = create_fallback_signal(price_data)

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        # 信号统计
        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        print(f"信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)")

        # 信号连续性检查
        if len(signal_history) >= 3:
            last_three = [s['signal'] for s in signal_history[-3:]]
            if len(set(last_three)) == 1:
                print(f"⚠️ 注意：连续3次{signal_data['signal']}信号")

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return create_fallback_signal(price_data)


def execute_intelligent_trade(signal_data, price_data):
    """执行智能交易 - OKX版本（支持同方向加仓减仓）"""
    global position
    did_reverse = False

    current_position = get_current_position()
    require_high_conf = TRADE_CONFIG.get('require_high_confidence_entry', True)
    print(f"当前持仓: {current_position}")

    # 无持仓时仅接受高信心开仓信号
    if (
        require_high_conf
        and not current_position
        and signal_data['signal'] in {'BUY', 'SELL'}
        and signal_data['confidence'] != 'HIGH'
    ):
        print("🔒 当前无持仓，仅高信心信号才允许开仓，跳过执行")
        _record_reverse_close_event(False)
        return

    # 防止频繁反转的逻辑保持不变
    if current_position and signal_data['signal'] != 'HOLD':
        current_side = current_position['side']  # 'long' 或 'short'

        if signal_data['signal'] == 'BUY':
            new_side = 'long'
        elif signal_data['signal'] == 'SELL':
            new_side = 'short'
        else:
            new_side = None

        # 如果方向相反，需要高信心才执行
        if new_side != current_side:
            if require_high_conf and signal_data['confidence'] != 'HIGH':
                print(f"🔒 非高信心反转信号，保持现有{current_side}仓")
                _record_reverse_close_event(False)
                return

            if not _can_reverse_recently():
                print("🔒 近期有反手平仓，避免频繁反转")
                _record_reverse_close_event(False)
                return

    # 计算智能仓位
    position_size = calculate_intelligent_position_v2(signal_data, price_data, current_position)
    if not position_size or position_size <= 0:
        print("⚠️ 目标仓位不可行（低于最小张数或保证金/费用不足），跳过执行")
        _record_reverse_close_event(False)
        return

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"智能仓位: {position_size:.2f} 张")
    print(f"理由: {signal_data['reason']}")
    # print(f"当前持仓: {current_position}")

    # 风险管理
    if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
        print("⚠️ 低信心信号，跳过执行")
        _record_reverse_close_event(False)
        return

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        _record_reverse_close_event(False)
        return

    try:
        # 执行交易逻辑 - 支持同方向加仓减仓
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                # 先检查空头持仓是否真实存在且数量正确
                if current_position['size'] > 0:
                    print(f"平空仓 {current_position['size']:.2f} 张并开多仓 {position_size:.2f} 张...")
                    # 平空仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        current_position['size'],
                        params={'reduceOnly': True}
                    )
                    time.sleep(1)
                    # 开多仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size
                )
                    did_reverse = True
                else:
                    print("⚠️ 检测到空头持仓但数量为0，直接开多仓")
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size
                    )

            elif current_position and current_position['side'] == 'long':
                # 同方向，检查是否需要调整仓位
                size_diff = position_size - current_position['size']

                if abs(size_diff) >= 0.01:  # 有可调整的差异
                    if size_diff > 0:
                        # 加仓
                        add_size = round(size_diff, 2)
                        print(
                            f"多仓加仓 {add_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'buy',
                            add_size
                        )
                    else:
                        # 减仓
                        reduce_size = round(abs(size_diff), 2)
                        print(
                            f"多仓减仓 {reduce_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'sell',
                            reduce_size,
                            params={'reduceOnly': True}
                        )
                else:
                    print(
                        f"已有多头持仓，仓位合适保持现状 (当前:{current_position['size']:.2f}, 目标:{position_size:.2f})")
            else:
                # 无持仓时开多仓
                print(f"开多仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    position_size
                )

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                # 先检查多头持仓是否真实存在且数量正确
                if current_position['size'] > 0:
                    print(f"平多仓 {current_position['size']:.2f} 张并开空仓 {position_size:.2f} 张...")
                    # 平多仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        current_position['size'],
                        params={'reduceOnly': True}
                    )
                    time.sleep(1)
                    # 开空仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size
                    )
                    did_reverse = True
                else:
                    print("⚠️ 检测到多头持仓但数量为0，直接开空仓")
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size
                    )

            elif current_position and current_position['side'] == 'short':
                # 同方向，检查是否需要调整仓位
                size_diff = position_size - current_position['size']

                if abs(size_diff) >= 0.01:  # 有可调整的差异
                    if size_diff > 0:
                        # 加仓
                        add_size = round(size_diff, 2)
                        print(
                            f"空仓加仓 {add_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'sell',
                            add_size
                        )
                    else:
                        # 减仓
                        reduce_size = round(abs(size_diff), 2)
                        print(
                            f"空仓减仓 {reduce_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'buy',
                            reduce_size,
                            params={'reduceOnly': True}
                        )
                else:
                    print(
                        f"已有空头持仓，仓位合适保持现状 (当前:{current_position['size']:.2f}, 目标:{position_size:.2f})")
            else:
                # 无持仓时开空仓
                print(f"开空仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    position_size
                )

        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            _record_reverse_close_event(False)
            return

        print("智能交易执行成功")
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")
        _record_reverse_close_event(did_reverse)

    except Exception as e:
        print(f"交易执行失败: {e}")

        # 如果是持仓不存在的错误，尝试直接开新仓
        if "don't have any positions" in str(e):
            print("尝试直接开新仓...")
            try:
                if signal_data['signal'] == 'BUY':
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size
                    )
                elif signal_data['signal'] == 'SELL':
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size
                    )
                print("直接开仓成功")
            except Exception as e2:
                print(f"直接开仓也失败: {e2}")

        import traceback
        traceback.print_exc()
        _record_reverse_close_event(did_reverse)


def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            print(f"第{attempt + 1}次尝试失败，进行重试...")
            time.sleep(1)

        except Exception as e:
            print(f"第{attempt + 1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait


def trading_bot():
    # 等待到整点再执行
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        return

    print(f"{get_price_label()}: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        print("⚠️ 使用备用交易信号")

    # 3. 执行智能交易
    execute_intelligent_trade(signal_data, price_data)


def main():
    """主函数"""
    print(f"{get_human_pair()} OKX自动交易机器人启动成功！")
    print("融合技术指标策略 + OKX实盘接口")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用完整技术指标分析和持仓跟踪功能")

    # 启动时打印关键配置
    print_runtime_config()

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    print("执行频率: 每15分钟整点执行")

    # 循环执行（不使用schedule）
    while True:
        trading_bot()  # 函数内部会自己等待整点

        # 执行完后等待一段时间再检查（避免频繁循环）
        time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    main()
