#!/usr/bin/env python3
"""Insert grounding/think+action failure masks into a copied dataset.py.

FAIL_SPAN_ROUTE=grounding_think_action  -> GR (this protocol)
FAIL_SPAN_ROUTE=grounding_action        -> old GA (must not be used as GR)
unset / other                           -> keep the file's existing default
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "FAILCTRL_V3_GR_MASK"

NEW_FN = '''def _token_weights_for_response(
    tokenizer,
    response: str,
    think_t: str,
    g_t: str,
    a_t: str,
    perc_weight: float,
    reas_weight: float,
) -> tuple[list[int], list[float], list[int]]:
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc["offset_mapping"])
    g_span = _char_span_in_text(response, g_t) if g_t else None
    think_span = _char_span_in_text(response, think_t) if think_t else None
    a_span = _char_span_in_text(response, a_t) if a_t else None
    route = os.environ.get("FAIL_SPAN_ROUTE", "")
    weights: list[float] = []
    mask: list[int] = []
    neg_mode = float(perc_weight) < 0.0 or float(reas_weight) < 0.0
    # FAILCTRL_V3_GR_MASK
    if neg_mode and route == "grounding_think_action":
        flags = []
        for s, e in offsets:
            if e <= s:
                flags.append("x")
                continue
            in_g = g_span is not None and s < g_span[1] and e > g_span[0]
            in_th = think_span is not None and s < think_span[1] and e > think_span[0]
            in_a = a_span is not None and s < a_span[1] and e > a_span[0]
            if in_g and not (in_th or in_a):
                flags.append("g")
            elif (in_th or in_a) and not in_g:
                flags.append("r")
            else:
                flags.append("x")  # format / overlap / neither
        n_g = flags.count("g")
        n_r = flags.count("r")
        for fl in flags:
            if fl == "g" and n_g > 0:
                weights.append(float(perc_weight) / n_g)
                mask.append(1)
            elif fl == "r" and n_r > 0:
                weights.append(float(reas_weight) / n_r)
                mask.append(1)
            else:
                weights.append(0.0)
                mask.append(0)
        return list(enc["input_ids"]), weights, mask
    if neg_mode and route == "grounding_action":
        n_g = n_a = 0
        flags = []
        for s, e in offsets:
            if e <= s:
                flags.append("x")
                continue
            in_g = g_span is not None and s < g_span[1] and e > g_span[0]
            in_a = a_span is not None and s < a_span[1] and e > a_span[0]
            if in_g and not in_a:
                flags.append("g"); n_g += 1
            elif in_a and not in_g:
                flags.append("a"); n_a += 1
            else:
                flags.append("x")
        for fl in flags:
            if fl == "g" and n_g:
                weights.append(float(perc_weight) / n_g); mask.append(1)
            elif fl == "a" and n_a:
                weights.append(float(reas_weight) / n_a); mask.append(1)
            else:
                weights.append(0.0); mask.append(0)
        return list(enc["input_ids"]), weights, mask
    neg_w = min(float(perc_weight), float(reas_weight))
    for s, e in offsets:
        if e <= s:
            weights.append(0.0)
            mask.append(0)
            continue
        if neg_mode:
            in_a = a_span is not None and s < a_span[1] and e > a_span[0]
            if in_a:
                weights.append(neg_w)
                mask.append(1)
            else:
                weights.append(0.0)
                mask.append(0)
            continue
        use_perc = g_span is not None and s < g_span[1] and e > g_span[0]
        w = float(perc_weight) if use_perc else float(reas_weight)
        if think_span is None and a_span is None and g_span is None:
            w = float(reas_weight)
        weights.append(w)
        mask.append(1)
    return list(enc["input_ids"]), weights, mask
'''


def patch(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        print("already patched", path)
        return
    if "import os" not in text.split("from __future__")[0] and "import os" not in text[:400]:
        text = text.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n", 1)
    start = text.index("def _token_weights_for_response(")
    end = text.index("def _image_token_counts(")
    text = text[:start] + NEW_FN + "\n\n" + text[end:]
    path.write_text(text)
    print("patched dataset GR masks", path)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
