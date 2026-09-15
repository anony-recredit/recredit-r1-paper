#!/usr/bin/env python3
"""Patch a copied trainer.py for failctrl_v3 dual-stream UL and max_steps.

Does not touch the live CE trainer used by n781 eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "FAILCTRL_V3_DUAL_STREAM"


def patch(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        print("already patched", path)
        return

    # max_steps: honor trainer.max_steps / FAILCTRL_MAX_STEPS after total_steps is set
    needle = "        self.total_steps = total_steps\n"
    inject = """        self.total_steps = total_steps
        # FAILCTRL_V3_DUAL_STREAM: optional shorter schedule (pilots use 0.25 U).
        _ms = os.environ.get("FAILCTRL_MAX_STEPS")
        if _ms:
            self.total_steps = min(self.total_steps, int(_ms))
            total_steps = self.total_steps
            warmup = int(float(cfg.optim.warmup_steps_ratio) * total_steps)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda s, warmup=warmup, total_steps=total_steps: (
                    min(1.0, (s + 1) / max(1, warmup))
                    if s < warmup
                    else max(0.1, 1.0 - (s - warmup) / max(1, total_steps - warmup))
                ),
            )
            if self.rank == 0:
                print("FAILCTRL_MAX_STEPS", self.total_steps)
"""
    if needle not in text:
        raise SystemExit("cannot find total_steps assignment")
    text = text.replace(needle, inject, 1)

    # Dual-stream failure loader after train_loader construction.
    loader_n = """        if self.rank == 0:
            print(OmegaConf.to_yaml(cfg))
            print(f"train={len(self.train_ds)} val={len(self.val_ds)} steps/epoch={steps_per_epoch}")
"""
    loader_i = """        self.fail_loader = None
        self.fail_iter = None
        fail_files = os.environ.get("FAILCTRL_FAIL_FILES")
        if fail_files:
            self.fail_ds = RecreditOTADataset(fail_files, tokenizer, processor, cfg.data)
            self.fail_sampler = DistributedSampler(
                self.fail_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
            )
            self.fail_loader = DataLoader(
                self.fail_ds,
                batch_size=int(cfg.data.train_batch_size),
                sampler=self.fail_sampler,
                num_workers=int(cfg.data.num_workers),
                pin_memory=True,
                drop_last=True,
                collate_fn=collate,
            )
            if self.rank == 0:
                print("FAILCTRL fail stream", len(self.fail_ds), "steps/epoch_fail", len(self.fail_loader))
        if self.rank == 0:
            print(OmegaConf.to_yaml(cfg))
            print(f"train={len(self.train_ds)} val={len(self.val_ds)} steps/epoch={steps_per_epoch}")
"""
    if loader_n not in text:
        raise SystemExit("cannot find train size print")
    text = text.replace(loader_n, loader_i, 1)

    # UL + separate pos/fail reduction overlay at start of _forward_loss stats
    old_mean = """        denom = shift_m.sum().clamp_min(1.0)
        loss = (token_ce * shift_w * shift_m).sum() / denom
"""
    # Remote trainer may already have an unlikelihood branch. Prefer inserting
    # a v3 override after token_ce is computed, before the default reduction.
    v3_loss = '''        # FAILCTRL_V3_DUAL_STREAM reduction: success CE and failure UL are separate.
        if os.environ.get("FAILCTRL_V3", "0") == "1":
            log_p = (-token_ce).float()
            p = torch.exp(log_p).clamp(0.0, 1.0)
            one_minus = torch.clamp(-torch.expm1(log_p), min=1e-6)
            ul = -torch.log(one_minus)
            pos = (shift_w > 0) & (shift_m > 0)
            neg = (shift_w < 0) & (shift_m > 0)
            pos_den = pos.float().sum().clamp_min(1.0)
            l_pos = (token_ce.float() * shift_w.float() * pos.float()).sum() / pos_den
            # weights already store beta_j = m alpha / n_span, so sum(|w| UL) = m (a_g U_g + a_r U_r)
            per = (ul * shift_w.abs().float() * neg.float()).sum(dim=-1)
            n_fail = (neg.any(dim=-1)).float().sum().clamp_min(1.0)
            l_fail = per.sum() / n_fail
            if not neg.any():
                loss = l_pos
            elif not pos.any():
                loss = l_fail
            else:
                loss = l_pos + l_fail
            denom = shift_m.sum().clamp_min(1.0)
            stats = {
                "loss": loss.detach(),
                "loss_pos": l_pos.detach(),
                "loss_fail": l_fail.detach(),
                "mean_abs_w": (shift_w.abs() * shift_m).sum().detach() / denom.detach(),
                "n_tok": denom.detach(),
                "n_fail_tok": neg.float().sum().detach(),
            }
            return loss, stats
        denom = shift_m.sum().clamp_min(1.0)
        loss = (token_ce * shift_w * shift_m).sum() / denom
'''
    if old_mean not in text:
        # already-unlikelihood trainer: still inject before first denom assignment in _forward_loss
        idx = text.find("        # L = mean_t [ w_t * CE_t ]")
        if idx < 0:
            idx = text.find("        denom = shift_m.sum().clamp_min(1.0)\n        loss = (token_ce")
        if idx < 0:
            raise SystemExit("cannot find CE reduction")
        # find that exact old_mean occurrence after token_ce
        pos = text.find("        denom = shift_m.sum().clamp_min(1.0)", text.find("token_ce = F.cross_entropy"))
        if pos < 0:
            raise SystemExit("cannot find denom after token_ce")
        # replace only this first occurrence in _forward_loss
        end = text.find("\n", pos)
        # replace two lines
        chunk_end = text.find("\n", text.find("\n", pos + 1) + 1)
        text = text[:pos] + v3_loss + text[chunk_end + 1 :]
    else:
        text = text.replace(old_mean, v3_loss, 1)

    # In the train loop, add a failure microbatch after the success one.
    # Find `loss, stats = self._forward_loss(micro)` first occurrence in fit/train
    hook = "                loss, stats = self._forward_loss(micro)\n"
    hook_i = """                loss, stats = self._forward_loss(micro)
                if self.fail_loader is not None:
                    if self.fail_iter is None:
                        self.fail_iter = iter(self.fail_loader)
                    try:
                        fbatch = next(self.fail_iter)
                    except StopIteration:
                        if hasattr(self, "fail_sampler"):
                            self.fail_sampler.set_epoch(epoch + 1000)
                        self.fail_iter = iter(self.fail_loader)
                        fbatch = next(self.fail_iter)
                    for fmicro in self._split_micro(fbatch):
                        floss, fstats = self._forward_loss(fmicro)
                        loss = loss + floss
                        stats = {**stats, "loss_fail_step": fstats.get("loss", floss.detach())}
"""
    if hook not in text:
        raise SystemExit("cannot find _forward_loss call in train loop")
    text = text.replace(hook, hook_i, 1)

    # Stop after max_steps inside epoch loop if set
    epoch_for = "        for epoch in range(int(self.cfg.trainer.total_epochs)):\n"
    if epoch_for in text:
        text = text.replace(
            epoch_for,
            epoch_for + "            if getattr(self, '_global_step', 0) >= int(self.total_steps):\n                break\n",
            1,
        )

    path.write_text(text)
    print("patched trainer v3", path)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
