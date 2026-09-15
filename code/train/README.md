# Recredit-R1 training recipe (verl-style)

Two-stage training from the paper, implemented as a verl FSDP recipe:

| Stage | Paper objective | Implementation |
|-------|-----------------|----------------|
| I geometric warm-start | \(\mathcal{L}_I=-\sum_t q_t[\log\pi_P(g_t\mid o_t)+\log\pi_R(\textit{think}_t,a_t\mid o_t,g_t)]\) | token-weighted SFT; all response tokens × \(q_t\) |
| II outcome-guided recrediting | \(\mathcal{L}_{II}=-\sum_t(A_t^{\mathrm{perc}}\log\pi_P+ \hat A_t^{\mathrm{reas}}\log\pi_R)\) | same trainer; grounding span × \(A^{\mathrm{perc}}\), thought+action × \(\hat A^{\mathrm{reas}}=\mu_t m_t A^{\mathrm{reas}}\) |

Backbone: **Qwen2.5-VL-7B-Instruct** (VLM required for RGB OTA). Framework: [verl-project/verl](https://github.com/verl-project/verl) `v0.4.1`.

## Suggested layout

```
${RECREDIT_ROOT}/
  miniconda3/          env `recredit`
  code/verl/           cloned framework
  code/recredit_r1/    this recipe
  models/Qwen2.5-VL-7B-Instruct/
  data/raw/            unpacked recredit.v1 pack (JSON + RGB)
  data/parquet/        trainer-format rows
  ckpts/stage1|stage2/
```

## Commands

```bash
# 1) framework + deps + model
bash code/train/setup_remote.sh

# 2) unpack pack + parquet
bash code/train/prepare_data.sh /path/to/pack.tar.gz

# 3) Stage I then Stage II (set NPROC to your GPU count)
bash code/train/run_stage1.sh
bash code/train/run_stage2.sh
```

Floorplans are held out by scene for val. Stage I keeps short/mid-horizon clips (`max-horizon=8`).
