"""Recredit-R1 multimodal dataset for verl FSDP SFT.

Each row is one OTA step: RGB + instruction prompt → thought/action response.
Token weights implement paper Stage I (q_t) and Stage II (A_perc / A_reas_hat)
on grounding vs thought+action spans.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _char_span_in_text(haystack: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    i = haystack.find(needle)
    if i < 0:
        return None
    return i, i + len(needle)


def _token_weights_for_response(
    tokenizer,
    response: str,
    think_t: str,
    g_t: str,
    a_t: str,
    perc_weight: float,
    reas_weight: float,
) -> tuple[list[int], list[float], list[int]]:
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    g_span = _char_span_in_text(response, g_t) if g_t else None
    think_span = _char_span_in_text(response, think_t) if think_t else None
    a_span = _char_span_in_text(response, a_t) if a_t else None

    weights: list[float] = []
    mask: list[int] = []
    for (s, e) in offsets:
        if e <= s:
            weights.append(0.0)
            mask.append(0)
            continue
        use_perc = False
        if g_span is not None and s < g_span[1] and e > g_span[0]:
            use_perc = True
        w = float(perc_weight) if use_perc else float(reas_weight)
        # thought/action empty: still supervise the whole assistant turn with reas weight
        if think_span is None and a_span is None and g_span is None:
            w = float(reas_weight)
        weights.append(w)
        mask.append(1)
    return ids, weights, mask


class RecreditOTADataset(Dataset):
    def __init__(self, parquet_files, tokenizer, processor, config):
        if isinstance(parquet_files, (str, Path)):
            parquet_files = [parquet_files]
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = int(config.get("max_length", 2048))
        self.truncation = config.get("truncation", "right")
        self.max_image_side = int(config.get("max_image_side", 448))
        frames = [pd.read_parquet(p) for p in parquet_files]
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > self.max_image_side:
            scale = self.max_image_side / m
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        response = str(row["response"])
        image_path = str(row["image"])
        image = self._load_image(image_path)

        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        messages_prompt = [{"role": "user", "content": user_content}]
        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]

        prompt_text = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        prompt_ids = prompt_inputs["input_ids"][0]
        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        pixel_values = full_inputs["pixel_values"]
        image_grid_thw = full_inputs["image_grid_thw"]

        prompt_len = int(prompt_ids.shape[0])
        resp_ids, resp_w, resp_m = _token_weights_for_response(
            self.tokenizer,
            response,
            str(row.get("think_t") or ""),
            str(row.get("g_t") or ""),
            str(row.get("a_t") or ""),
            float(row["perc_weight"]),
            float(row["reas_weight"]),
        )

        seq_len = int(input_ids.shape[0])
        loss_weight = torch.zeros(seq_len, dtype=torch.float32)
        loss_mask = torch.zeros(seq_len, dtype=torch.long)
        # Align independently tokenized response weights to the assistant suffix.
        suffix_len = max(0, seq_len - prompt_len)
        n = min(suffix_len, len(resp_w))
        if n > 0:
            # Place weights at the end of the suffix so chat-template prefix tokens
            # (role headers) stay unweighted; trailing template tokens get 0.
            start = prompt_len + max(0, suffix_len - len(resp_w))
            end = start + n
            loss_weight[start:end] = torch.tensor(resp_w[:n], dtype=torch.float32)
            loss_mask[start:end] = torch.tensor(resp_m[:n], dtype=torch.long)

        if seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_weight = loss_weight[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            else:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_weight = loss_weight[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            seq_len = self.max_length

        position_ids = torch.arange(seq_len, dtype=torch.long)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "position_ids": position_ids,
            "loss_weight": loss_weight,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "q_t": torch.tensor(float(row["q_t"]), dtype=torch.float32),
            "R_L2": torch.tensor(int(row["R_L2"]), dtype=torch.long),
        }


def collate_recredit(features: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    def pad_1d(key, padding_value, dtype=None):
        seqs = [f[key] for f in features]
        out = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
        if dtype is not None:
            out = out.to(dtype)
        return out

    pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
    image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
    return {
        "input_ids": pad_1d("input_ids", pad_token_id, torch.long),
        "attention_mask": pad_1d("attention_mask", 0, torch.long),
        "position_ids": pad_1d("position_ids", 0, torch.long),
        "loss_weight": pad_1d("loss_weight", 0.0, torch.float32),
        "loss_mask": pad_1d("loss_mask", 0, torch.long),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "q_t": torch.stack([f["q_t"] for f in features]),
        "R_L2": torch.stack([f["R_L2"] for f in features]),
    }
