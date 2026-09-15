#!/usr/bin/env python3
"""
Recredit-R1 two-stage trainer on verl FSDP.

Stage I:  L_I  = -sum_t q_t [log pi_P(g_t|o_t) + log pi_R(think_t,a_t|o_t,g_t)]
Stage II: L_II = -sum_t [A_perc log pi_P(g_t|o_t) + A_reas_hat log pi_R(...)]

Token weights are baked into the parquet by convert_recredit_to_parquet.py.
This trainer multiplies token-wise CE by those weights (signed A_t allowed).
"""
from __future__ import annotations

import logging
import os
from functools import partial

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from dataset import RecreditOTADataset, collate_recredit

logger = logging.getLogger(__name__)


def _init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def _load_vlm(model_path: str, attn_impl: str, torch_dtype):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    return model


def _wrap_policy(model):
    layer_cls = set()
    for name in (
        "Qwen2_5_VLDecoderLayer",
        "Qwen2VLDecoderLayer",
        "Qwen2_5_VLVisionBlock",
        "Qwen2VLVisionBlock",
        "Qwen2DecoderLayer",
    ):
        for mod in model.modules():
            if mod.__class__.__name__ == name:
                layer_cls.add(mod.__class__)
    if not layer_cls:
        # fallback: wrap every module with children named "self_attn"
        for mod in model.modules():
            if hasattr(mod, "self_attn"):
                layer_cls.add(mod.__class__)
                break
    return partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


class RecreditTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.local_rank, self.rank, self.world_size = _init_dist()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.manual_seed(int(cfg.trainer.seed) + self.rank)

        processor = AutoProcessor.from_pretrained(cfg.model.path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.processor = processor
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

        self.train_ds = RecreditOTADataset(cfg.data.train_files, tokenizer, processor, cfg.data)
        self.val_ds = RecreditOTADataset(cfg.data.val_files, tokenizer, processor, cfg.data)

        self.train_sampler = DistributedSampler(
            self.train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
        )
        collate = partial(collate_recredit, pad_token_id=self.pad_id)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=int(cfg.data.train_batch_size),
            sampler=self.train_sampler,
            num_workers=int(cfg.data.num_workers),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate,
        )
        self.val_sampler = DistributedSampler(
            self.val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=int(cfg.data.micro_batch_size_per_gpu),
            sampler=self.val_sampler,
            num_workers=int(cfg.data.num_workers),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
        )

        dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16
        attn = cfg.model.attn_implementation
        try:
            model = _load_vlm(cfg.model.path, attn, dtype)
        except Exception as e:
            logger.warning("attn %s failed (%s); falling back to sdpa", attn, e)
            model = _load_vlm(cfg.model.path, "sdpa", dtype)

        if cfg.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        mesh = init_device_mesh("cuda", mesh_shape=(self.world_size,), mesh_dim_names=("fsdp",))
        self.model = FSDP(
            model,
            auto_wrap_policy=_wrap_policy(model),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=self.local_rank,
            use_orig_params=True,
            device_mesh=mesh,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.optim.lr),
            betas=tuple(cfg.optim.betas),
            weight_decay=float(cfg.optim.weight_decay),
        )
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = int(cfg.trainer.total_epochs) * steps_per_epoch
        warmup = int(float(cfg.optim.warmup_steps_ratio) * total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: min(1.0, (s + 1) / max(1, warmup)) if s < warmup else max(0.1, 1.0 - (s - warmup) / max(1, total_steps - warmup)),
        )
        self.total_steps = total_steps
        self.micro = int(cfg.data.micro_batch_size_per_gpu)
        self.accum = max(1, int(cfg.data.train_batch_size) // self.micro)

        if self.rank == 0:
            print(OmegaConf.to_yaml(cfg))
            print(f"train={len(self.train_ds)} val={len(self.val_ds)} steps/epoch={steps_per_epoch}")

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict]:
        device = self.device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        if pixel_values.dtype != torch.bfloat16 and self.cfg.model.precision == "bf16":
            pixel_values = pixel_values.bfloat16()
        image_grid_thw = batch["image_grid_thw"].to(device)
        loss_weight = batch["loss_weight"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_w = loss_weight[:, 1:].contiguous()
        shift_m = loss_mask[:, 1:].contiguous().float()

        vocab = shift_logits.size(-1)
        token_ce = F.cross_entropy(
            shift_logits.float().view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        # L = mean_t [ w_t * CE_t ] over supervised tokens; w_t may be signed (Stage II).
        denom = shift_m.sum().clamp_min(1.0)
        loss = (token_ce * shift_w * shift_m).sum() / denom

        stats = {
            "loss": loss.detach(),
            "mean_abs_w": (shift_w.abs() * shift_m).sum().detach() / denom.detach(),
            "n_tok": denom.detach(),
        }
        return loss, stats

    def _split_micro(self, batch):
        bsz = batch["input_ids"].size(0)
        if bsz <= self.micro:
            yield batch
            return
        keys = list(batch.keys())
        for i in range(0, bsz, self.micro):
            sl = slice(i, i + self.micro)
            micro = {}
            for k in keys:
                v = batch[k]
                if k in ("pixel_values", "image_grid_thw"):
                    # pixel_values is concatenated patches, not batch-aligned 1:1 with samples
                    # when micro-batching, re-slice by image_grid_thw rows (one image per sample).
                    continue
                micro[k] = v[sl]
            # one image / sample in this dataset
            micro["image_grid_thw"] = batch["image_grid_thw"][sl]
            # pixel_values rows correspond to sum of grid patches per image
            grids = batch["image_grid_thw"]
            starts = [0]
            for g in grids:
                t, h, w = int(g[0]), int(g[1]), int(g[2])
                starts.append(starts[-1] + t * h * w)
            s0, s1 = starts[i], starts[min(i + self.micro, len(starts) - 1)]
            micro["pixel_values"] = batch["pixel_values"][s0:s1]
            yield micro

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = torch.zeros(2, device=self.device)
        for batch in self.val_loader:
            loss, stats = self._forward_loss(batch)
            total[0] += stats["loss"] * stats["n_tok"]
            total[1] += stats["n_tok"]
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        self.model.train()
        return float((total[0] / total[1].clamp_min(1)).item())

    def save(self, tag: str):
        out = os.path.join(self.cfg.trainer.save_dir, tag)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            cpu_state = self.model.state_dict()
        if self.rank == 0:
            os.makedirs(out, exist_ok=True)
            torch.save(cpu_state, os.path.join(out, "fsdp_full.pt"))
            self.processor.save_pretrained(out)
            OmegaConf.save(self.cfg, os.path.join(out, "config.yaml"))
            # HuggingFace export so Stage II can resume from Stage I
            try:
                bare = _load_vlm(self.cfg.model.path, "sdpa", torch.float32)
                missing, unexpected = bare.load_state_dict(cpu_state, strict=False)
                bare.save_pretrained(out)
                print("saved HF", out, "missing", len(missing), "unexpected", len(unexpected))
            except Exception as e:
                print("HF export skipped:", e)
        dist.barrier()

    def fit(self):
        global_step = 0
        for epoch in range(int(self.cfg.trainer.total_epochs)):
            self.train_sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            running = 0.0
            nseen = 0
            for batch in self.train_loader:
                micros = list(self._split_micro(batch))
                n_micro = len(micros)
                for mi, micro in enumerate(micros):
                    loss, stats = self._forward_loss(micro)
                    (loss / n_micro).backward()
                    running += float(stats["loss"].item())
                    nseen += 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.optim.clip_grad))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if self.rank == 0 and global_step % int(self.cfg.trainer.log_freq) == 0:
                    print(
                        f"epoch={epoch} step={global_step}/{self.total_steps} "
                        f"loss={running / max(1, nseen):.4f} lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    running, nseen = 0.0, 0
                if int(self.cfg.trainer.save_freq) > 0 and global_step % int(self.cfg.trainer.save_freq) == 0:
                    self.save(f"global_step_{global_step}")
            if int(self.cfg.trainer.test_freq) != 0:
                val = self.validate()
                if self.rank == 0:
                    print(f"epoch={epoch} val_loss={val:.4f}")
            self.save(f"epoch_{epoch}")
        self.save("final")
        dist.barrier()
        dist.destroy_process_group()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    RecreditTrainer(cfg).fit()


if __name__ == "__main__":
    main()
