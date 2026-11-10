"""Automatic quality gates.

The goal is to replace human review: every failure mode the pipeline can detect
must either (a) trigger an automatic retry/escalation, or (b) be recorded as a
machine-readable warning with a confidence score.
"""

from __future__ import annotations

import re
from typing import Any

from .vendor.document import finalize_document
from .vendor.otsl import convert_otsl_to_html
from .vendor.repetition_guard import analyze_repetition

OTSL_RE = re.compile(r"<(?:fcel|ecel|nl|lcel|ucel|xcel)>")


def warning(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, **extra}


def finalize(text: str) -> str:
    """Upstream whole-page cleanup: repetition truncation, math delimiters, OTSL→HTML."""
    try:
        return finalize_document(text)
    except Exception:
        return text


def convert_table(otsl: str) -> tuple[str, bool]:
    """Return (html, ok). Never raise: raw OTSL is kept when conversion fails."""
    try:
        html = convert_otsl_to_html(otsl)
        return (html if html else otsl), bool(html)
    except Exception:
        return otsl, False


def assess(
    text: str,
    finish_reason: str | None = None,
    *,
    mode: str = "page",
    min_chars: int = 1,
) -> dict[str, Any]:
    """Score one OCR response and decide whether an automatic retry is warranted."""
    text = text or ""
    warnings: list[dict[str, Any]] = []
    score = 1.0

    if len(text.strip()) < min_chars:
        warnings.append(warning("empty_output", "high",
                                "Model returned no usable text for this region/page."))
        score = 0.0

    analysis = analyze_repetition(text, finish_reason)
    if analysis["suspicious"]:
        warnings.append(warning("repetition", "medium",
                                "Repetitive output detected; an automatic retry will be attempted.",
                                reasons=analysis["reasons"]))
        score -= 0.5

    if finish_reason == "length":
        warnings.append(warning("truncated", "high",
                                "Output hit the token budget and was truncated."))
        score -= 0.4

    if OTSL_RE.search(text):
        _, ok = convert_table(text)
        if not ok:
            warnings.append(warning("table_parse_failed", "medium",
                                    "Table output could not be converted; raw OTSL kept."))
            score -= 0.2

    score = max(0.0, min(1.0, score))
    return {
        "warnings": warnings,
        "score": round(score, 3),
        "suspicious": bool(analysis["suspicious"]),
        "needs_retry": bool(analysis["suspicious"]) or finish_reason == "length"
        or len(text.strip()) < min_chars,
        "analysis": analysis,
        "mode": mode,
    }


def _otsu_threshold(arr) -> int:
    import numpy as np
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    sum_all = float(np.dot(np.arange(256), hist))
    weight_bg = 0.0
    sum_bg = 0.0
    best_var, best_t = 0.0, 127
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def ink_ratio(image) -> float:
    """Fraction of foreground pixels using an adaptive (Otsu) threshold.

    A fixed cutoff is fooled by dark paper/textured backgrounds: a photo of
    calligraphy can read as "100% ink" while containing very little text.
    """
    try:
        import numpy as np
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
        if arr.size == 0 or int(arr.max()) - int(arr.min()) < 40:
            return 0.0                     # flat page: no real foreground
        threshold = _otsu_threshold(arr)
        return float((arr <= threshold).mean())
    except Exception:
        return 0.0


def coverage_warning(text: str, image) -> dict | None:
    """Catch the failure mode where the model returns only a title or a line."""
    ink = ink_ratio(image)
    chars = len(re.sub(r"\s", "", text or ""))
    # Deliberately conservative: short pages are legitimate (excerpts, title
    # pages, calligraphy). Only flag a near-empty response on a busy page.
    if chars < 10 and ink > 0.15:
        return warning("low_coverage", "medium",
                       f"Page looks dense (ink {ink:.1%}) but output has only {chars} chars.",
                       ink=round(ink, 4), chars=chars)
    return None


def parse_kie(raw: str, fields: list[str] | None = None) -> tuple[dict | None, str]:
    """Extract the first JSON object from a KIE response."""
    import json
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        return None, "no_json_object"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "not_an_object"
    if fields:
        data = {f: data.get(f, "") for f in fields}
    return data, ""


def _digits(value: str) -> str:
    return re.sub(r"[^\d]", "", value or "")


_DATE_RE = re.compile(r"^\s*(\d{2,4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\s*$")


def _date_parts(value: str) -> list[str] | None:
    match = _DATE_RE.match(value or "")
    return [match.group(1), match.group(2), match.group(3)] if match else None


def _part_in_text(part: str, text_digits: str) -> bool:
    if part in text_digits:
        return True
    if len(part) == 4 and part[-2:] in text_digits:      # 2017 → 17 (YY-MM-DD)
        return True
    stripped = part.lstrip("0")
    return bool(stripped) and stripped in text_digits


def ground_fields(data: dict, page_text: str) -> dict[str, dict]:
    """Hallucination check: every extracted value must appear in the OCR text.

    Numeric values are compared digit-by-digit (so 1,234.56 matches 1234.56 and
    2024-01-05 matches 2024年1月5日); text values are compared after whitespace
    stripping.
    """
    text_digits = _digits(page_text)
    text_flat = re.sub(r"\s+", "", page_text or "")
    results: dict[str, dict] = {}
    for key, value in (data or {}).items():
        if not isinstance(value, str) or not value.strip():
            results[key] = {"ok": False, "reason": "empty"}
            continue
        parts = _date_parts(value)
        if parts and all(_part_in_text(p, text_digits) for p in parts):
            # ISO-normalized date extracted from e.g. 18-10-17 is legitimate.
            results[key] = {"ok": True, "reason": "date_components_present"}
            continue
        digits = _digits(value)
        if len(digits) >= 3:
            ok = digits in text_digits
        else:
            ok = re.sub(r"\s+", "", value) in text_flat
        results[key] = {"ok": ok, "reason": "" if ok else "not_found_in_ocr_text"}
    return results


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numeric_tokens(text: str) -> list[str]:
    return NUM_RE.findall(text or "")


def cross_check_numbers(primary: str, secondary: str, *, limit: int = 40) -> dict[str, Any]:
    """Compare numeric tokens between two independent readings.

    Used by the automatic second-pass verifier for money/date/id-heavy documents.
    Returns the tokens that disagree; an empty list means agreement.
    """
    a = numeric_tokens(primary)[:limit]
    b = numeric_tokens(secondary)[:limit]
    if not a and not b:
        return {"ok": True, "mismatches": [], "compared": 0}
    mismatches = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.replace(",", "") != y.replace(",", ""):
            mismatches.append({"index": i, "primary": x, "secondary": y})
    if len(a) != len(b):
        mismatches.append({"index": min(len(a), len(b)),
                           "primary": f"{len(a)} tokens", "secondary": f"{len(b)} tokens"})
    total = max(len(a), len(b), 1)
    return {"ok": not mismatches, "mismatches": mismatches[:10],
            "compared": total, "agreement": round(1 - len(mismatches) / total, 3)}
