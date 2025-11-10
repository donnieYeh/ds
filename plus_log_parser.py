import re
from datetime import datetime
from typing import List, Dict, Any
import ast
import json


HEADER_LINE = "============================================================"

# Matches leading timestamps like:
#  - 2025-11-06T09:00:11:
#  - 2025-11-06 09:00:11:
#  - 2025-11-06T09:00:11.123Z:
#  - 2025-11-06T09:00:11+08:00:
TS_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?:\s*")


def _strip_ts_prefix(line: str) -> str:
    return TS_PREFIX_RE.sub("", line, count=1)


def _safe_float(val: str):
    try:
        return float(val)
    except Exception:
        return None


def _parse_price(line: str):
    # ETH当前价格: $3,975.97
    m = re.search(r"\$\s*([0-9,]+\.?[0-9]*)", line)
    if m:
        return _safe_float(m.group(1).replace(",", ""))
    return None


def _parse_exec_time(line: str):
    # 执行时间: 2025-10-29 08:08:06
    ts = line.split("执行时间:", 1)[-1].strip()
    # Return raw string for display and also an iso if parsable
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return ts, dt.isoformat()
    except Exception:
        return ts, None


def _parse_position(text: str):
    try:
        return ast.literal_eval(text)
    except Exception:
        return text.strip()


def _parse_ds_json(ds_text: str):
    """Attempt to parse DeepSeek JSON even if formatting is slightly invalid."""
    if not ds_text:
        return None
    l = ds_text.find("{")
    r = ds_text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    snippet = ds_text[l:r + 1].strip()
    if not snippet:
        return None

    def _try_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    obj = _try_json(snippet)
    if obj is not None:
        return obj

    try:
        return ast.literal_eval(snippet)
    except Exception:
        pass

    cleaned = snippet.replace("'", '"')
    cleaned = re.sub(r'(\b\w+\b)\s*:', r'"\1":', cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return _try_json(cleaned)


def split_records(log_text: str) -> List[str]:
    """Split the full log into per-run records.

    Target pattern (observed):
    ============================================================
    执行时间: YYYY-MM-DD HH:MM:SS
    ============================================================
    <content lines...>
    [then another '============================================================' starting the next record]

    We capture from the '执行时间:' line inclusive until just before the next header that itself precedes
    another '执行时间:' line.
    """
    lines = log_text.splitlines()
    n = len(lines)
    i = 0
    out: List[str] = []
    while i < n:
        s = lines[i].strip()
        next_line = lines[i + 1] if i + 1 < n else ""
        next_no_ts = _strip_ts_prefix(next_line).strip()
        if s == HEADER_LINE and next_no_ts.startswith("执行时间:"):
            # Start of a record
            start_idx = i + 1  # include 执行时间行
            # Skip next header line if present immediately after exec_time
            j = start_idx + 1
            if j < n and lines[j].strip() == HEADER_LINE:
                j += 1
            # Collect until next header that starts a new record
            k = j
            while k < n:
                if lines[k].strip() == HEADER_LINE and (k + 1) < n and _strip_ts_prefix(lines[k + 1]).strip().startswith("执行时间:"):
                    break
                k += 1
            out.append("\n".join(lines[start_idx:k]).strip())
            i = k
            continue
        i += 1

    # Fallback: if nothing matched, fall back to splitting by 执行时间: lines only
    if not out:
        cur: List[str] = []
        for ln in lines:
            if _strip_ts_prefix(ln).strip().startswith("执行时间:"):
                if cur:
                    out.append("\n".join(cur).strip())
                    cur = []
                cur.append(ln)
            else:
                if cur:
                    cur.append(ln)
        if cur:
            out.append("\n".join(cur).strip())

    return [r for r in out if r]


def parse_plus_log(log_text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rec in split_records(log_text):
        lines = rec.splitlines()
        data: Dict[str, Any] = {
            "raw": rec,
        }
        # Extract fields
        i = 0
        # exec time
        if lines and _strip_ts_prefix(lines[0]).strip().startswith("执行时间:"):
            raw_ts, iso_ts = _parse_exec_time(_strip_ts_prefix(lines[0]))
            data["exec_time"] = raw_ts
            data["exec_time_iso"] = iso_ts
            i = 1
        # Scan rest
        # Capture DeepSeek block if present (brace-balanced after the marker)
        ds_raw = None
        brace_depth = 0
        in_ds = False
        ds_mode = None  # 'brace' or 'fenced'
        ds_lines: List[str] = []
        ds_fence_opened = False

        # Helper to finish ds block
        def finish_ds():
            nonlocal ds_raw, in_ds, ds_lines, brace_depth, ds_mode, ds_fence_opened
            if ds_lines:
                ds_raw = "\n".join(ds_lines).strip()
            in_ds = False
            brace_depth = 0
            ds_lines = []
            ds_mode = None
            ds_fence_opened = False

        while i < len(lines):
            line = lines[i]
            s = _strip_ts_prefix(line).strip()
            if s.startswith("ETH当前价格:"):
                data["eth_price"] = _parse_price(s)
            elif s.startswith("数据周期:"):
                data["timeframe"] = s.split(":", 1)[-1].strip()
            elif s.startswith("价格变化:"):
                data["price_change"] = s.split(":", 1)[-1].strip()
            elif s.startswith("DeepSeek原始回复:"):
                # everything after the colon could include a JSON-like block
                after = s.split(":", 1)[-1].lstrip()
                if after:
                    in_ds = True
                    ds_lines.append(after)
                    if after.strip().startswith("```"):
                        ds_mode = 'fenced'
                        ds_fence_opened = True
                    else:
                        ds_mode = 'brace'
                        brace_depth += after.count("{") - after.count("}")
                        if brace_depth <= 0:
                            finish_ds()
                else:
                    in_ds = True
                    ds_mode = None
                # continue
            elif in_ds:
                plain_line = _strip_ts_prefix(line)
                ds_lines.append(plain_line)
                stripped_line = plain_line.strip()
                if ds_mode == 'fenced':
                    # close on next fence line
                    if stripped_line.startswith("```") and ds_fence_opened:
                        finish_ds()
                    else:
                        ds_fence_opened = True  # we passed the opening line
                else:
                    # default to brace mode if not yet set
                    if ds_mode is None:
                        if stripped_line.startswith("```"):
                            ds_mode = 'fenced'
                            ds_fence_opened = True
                        else:
                            ds_mode = 'brace'
                            brace_depth = 0
                    if ds_mode == 'brace':
                        brace_depth += stripped_line.count("{") - stripped_line.count("}")
                        if brace_depth <= 0:
                            finish_ds()
            elif s.startswith("信号统计:"):
                data["signal_stats"] = s.split(":", 1)[-1].strip()
            elif s.startswith("⚠️") or s.startswith("⚠") or s.startswith("🔒") or s.startswith("❗") or s.startswith("ℹ"):
                data["warning"] = s
            elif s.startswith("🕒"):
                # clock/info lines (等待到整点等) — ignore so they don't override action
                pass
            elif s.startswith("交易信号:"):
                data["signal"] = s.split(":", 1)[-1].strip()
            elif s.startswith("信心程度:"):
                data["confidence"] = s.split(":", 1)[-1].strip()
            elif s.startswith("理由:"):
                data["reason"] = s.split(":", 1)[-1].strip()
            elif s.startswith("止损:"):
                data["stop_loss"] = _parse_price(s)
            elif s.startswith("止盈:"):
                data["take_profit"] = _parse_price(s)
            elif s.startswith("当前持仓:") or s.startswith("更新后持仓:"):
                pos_text = s.split(":", 1)[-1].strip()
                data["position"] = _parse_position(pos_text)
            elif s.startswith("根据账户余额调整下单数量为"):
                # 根据账户余额调整下单数量为 0.02 ETH (原始配置: 0.05 ETH)
                m = re.search(r"为\s+([0-9.]+)\s*ETH.*原始配置:\s*([0-9.]+)\s*ETH", s)
                if m:
                    data["adjusted_size"] = _safe_float(m.group(1))
                    data["original_size"] = _safe_float(m.group(2))
                data["sizing_note"] = s
            elif s:
                # Heuristic: capture last non-empty line as action/suggestion
                data["action"] = s

            i += 1

        if ds_raw:
            data["deepseek_raw"] = ds_raw
            obj = _parse_ds_json(ds_raw)
            if obj:
                data.setdefault("signal", obj.get("signal"))
                data.setdefault("confidence", obj.get("confidence"))
                data.setdefault("reason", obj.get("reason"))
                for key, target in (("stop_loss", "stop_loss"), ("take_profit", "take_profit")):
                    val = obj.get(key)
                    if val is None:
                        continue
                    try:
                        data.setdefault(target, float(val))
                    except Exception:
                        pass

        # Generate an id for record (prefer exec_time)
        rec_id = data.get("exec_time") or data.get("exec_time_iso") or str(len(items))
        data["id"] = rec_id

        items.append(data)

    # Sort by time if iso available; else keep order
    def sort_key(x):
        return x.get("exec_time_iso") or x.get("exec_time") or ""

    items.sort(key=sort_key)
    return items
