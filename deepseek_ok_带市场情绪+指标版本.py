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


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本
    recent_n = TRADE_CONFIG.get('recent_kline_count', 20)
    kline_text = f"【最近{recent_n}根{TRADE_CONFIG['timeframe']}K线数据(K线{recent_n}为最新数据)】\n"
    for i, kline in enumerate(price_data['kline_data'][-recent_n:]):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"    K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n  【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 获取情绪数据
    sentiment_data = get_sentiment_indicators_with_retry()
    # 简化情绪文本 多了没用
    if sentiment_data:
        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        sentiment_text = f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
    else:
        sentiment_text = "【市场情绪】数据暂不可用"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
    pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']:.2f} USDT" if current_pos else ""

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下{get_human_pair()} {TRADE_CONFIG['timeframe']}周期数据进行分析：

    【防频繁交易重要原则】
    1. **趋势持续性优先**: 不要因单根K线或短期波动改变整体趋势判断
    2. **持仓稳定性**: 除非趋势明确强烈反转，否则保持现有持仓方向
    3. **反转确认**: 需要至少2-3个技术指标同时确认趋势反转才改变信号
    4. **成本意识**: 减少不必要的仓位调整，每次交易都有成本

    【趋势与风险平衡原则】
    1. 趋势优先，但要识别“透支风险”：
        - 当短期与中期均线方向一致、价格沿多头方向运行时，可以优先考虑顺势交易。
        - 但如果此时多项信号同时指向“行情可能已经接近阶段尾声”（例如：价格连续创高但动能放缓、动量指标处于高位区间、价格多次触及上轨附近等），你需要降低做多信心，而不是简单视为更强信号。
    2. 请主动识别以下“可能透支”的组合特征（不限于固定阈值）：
        - 价格处于近期波动区间的上沿或明显高位；
        - 动量指标在高位但边际增量减弱（如MACD柱体缩短、RSI高位横盘等）；
        - 突破后缺乏持续跟进（如放量冲高后回落、上影线明显等）。
        遇到这些情况，你应更偏向：
            - 降低信号置信度；
            - 建议小仓或观望；
            - 给出“等待更好入场位置”的理由。
    3. 在趋势向上而无明显透支迹象时：
        - 你可以给出BUY信号，并根据技术结构和波动环境给出合理的置信度和仓位建议。
        - 不需要机械依赖某一个指标的单点阈值，而是综合评估多项信息的一致性与可持续性。
    4. 若技术信号之间存在明显冲突：
        - 例如：趋势看多，但多项信号提示可能见顶或动能衰减，
        - 优先选择更保守的方案（降低置信度、小仓或HOLD），并在理由中说明冲突点。

    【动能透支处理原则】

        你会收到一段“动能透支评估 - 系统辅助信息”，其中包含 level（none/mild/strong）以及参考信号说明。
        请按以下方式理解和使用（这是思考方向，而不是死规则）：

        - 若 level = "strong":
        - 优先考虑这是阶段性高风险区域；
        - 降低做多信心，倾向小仓或观望，而不是给出 HIGH 信心 BUY；
        - 如仍认为可以做多，必须在理由中清晰说明为何当前结构仍支持顺势参与。

        - 若 level = "mild":
        - 说明部分高位或动能放缓迹象，需要更谨慎评估；
        - 可以给出 BUY，但不应简单视为“无脑强势”，应考虑更合理的仓位与保护。

        - 若 level = "none":
        - 说明当前不存在明显透支信号，可以更专注于趋势与结构本身的判断。

        在任何情况下，请综合 K线结构、趋势、动量、布林带和情绪，不要因为“强势”或“单一信号”就给出激进决策。


    【交易指导原则 - 必须遵守】
    1. **技术分析主导** (权重60%)：趋势、支撑阻力、K线形态是主要依据
    2. **市场情绪辅助** (权重30%)：情绪数据用于验证技术信号，不能单独作为交易理由  
        - 情绪与技术同向 → 增强信号信心
        - 情绪与技术背离 → 以技术分析为主，情绪仅作参考
        - 情绪数据延迟 → 降低权重，以实时技术指标为准
    3. **风险管理** (权重10%)：考虑持仓、盈亏状况和止损位置
    7. **技术指标权重**:
        - 趋势(均线排列) > RSI > MACD > 布林带
        - 价格突破关键支撑/阻力位是重要信号 

    【智能仓位管理规则 - 必须遵守】
    1. **减少过度保守**：
       - 明确趋势中不要因轻微超买/超卖而过度HOLD
       - RSI在30-70区间属于健康范围，不应作为主要HOLD理由
       - 布林带位置在20%-80%属于正常波动区间

    2. **与趋势协调的仓位调整**：
       - 当技术与趋势大体一致且无明显透支迹象时，可以考虑适度提高仓位；
       - 当出现潜在透支或反转信号时，应自动降低仓位或保持观望；
       - 仓位调整应以【趋势与风险平衡原则】为前提，而非简单因为“强势”就加码。

    3. **突破交易信号**：
       - 价格突破关键阻力 + 成交量放大 → 高信心BUY
       - 价格跌破关键支撑 + 成交量放大 → 高信心SELL

    4. **持仓优化逻辑**：
       - 已有持仓且趋势延续 → 保持或BUY/SELL信号
       - 趋势明确反转 → 及时反向信号
       - 不要因为已有持仓而过度HOLD

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

    {kline_text}

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
    print(f"当前持仓: {current_position}")

    # 无持仓时仅接受高信心开仓信号
    if not current_position and signal_data['signal'] in {'BUY', 'SELL'} and signal_data['confidence'] != 'HIGH':
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
            if signal_data['confidence'] != 'HIGH':
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
