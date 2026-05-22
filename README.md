# Live Music Diffusion Models

[**Demo Page**](https://stephenbrade.github.io/lmdm-public/) | [**Paper**](https://arxiv.org/abs/2605.22717)

Training and inference code for Live Music Diffusion Models (LMDMs): streaming, autoregressive music diffusion models. Models
generate audio block-by-block over a sliding context window, supporting live generation. Huge shout-out to the [Stable Audio](https://github.com/Stability-AI/stable-audio-tools) folks, where this codebase draws heavy inspiration from.

This is our public facing code repo. For access to development code used during the project, please reach out to znovack@ucsd.edu or brade@mit.edu.

## Install

```bash
$ pip install .
```

Requires PyTorch 2.5+ (Flash / Flex Attention). Developed against Python 3.10.

## Models

Two attention regimes, each available as a plain finetune or as an ARC-forcing model:

| Config | Attention | Type |
| --- | --- | --- |
| `saos_encdec.json` | enc-dec (bidirectional context) | finetune |
| `saos_block_causal.json` | block-causal (sliding-window causal) | finetune |
| `saos_arc_forcing_encdec.json` | enc-dec | ARC-forcing |
| `saos_arc_forcing_block_causal.json` | block-causal | ARC-forcing |

Configs live in `stable_audio_tools/configs/model_configs/txt2audio/`.

## Training

```bash
python train.py \
    --model-config stable_audio_tools/configs/model_configs/txt2audio/<config>.json \
    --dataset-config <your_dataset>.json \
    --pretrained-ckpt-path /path/to/base.ckpt \
    --save-dir ./checkpoints \
    --batch-size 40 --precision 16-mixed --name <run-name>
```

Training should proceed in two stages:

1. **Finetune:** use `saos_encdec.json` or `saos_block_causal.json`. This mirrors standard diffusion finetuning and has the same overall memory bandwidth. Initialize this with your standard favorite music diffusion model (SAO, SAO-Small).
2.  **ARC-forcing:** use `saos_arc_forcing_encdec.json` or `saos_arc_forcing_block_causal.json`.
  ARC configs set `training.arc.self_forcing` and pull the teacher/discriminator from the
  base model; the attention regime is set by `training.inpainting.mask_kwargs.context_router_attention_pattern`. This should be initialized from your finetuned LMDM in the first step. Note that the memory bandwidth here will increase as a function of the rollout length, so plan accordingly.

See [train.sh](train.sh) for an end-to-end launch example. Training defaults are in [defaults.ini](defaults.ini).

## Inference

Streaming block-AR generation goes through
[`generate_diffusion_cond_blockar`](stable_audio_tools/inference/generation.py) — it denoises one
`block_size` block at a time over a sliding context window, optionally reusing a KV cache for
fast streaming. Set `context_router_attention_pattern` to match the model (`"enc-dec"` or
`"block-causal"`) and pass `use_kv_cache=True` for streaming.

A runnable end-to-end example (loading a checkpoint, building conditioning, calling the
function, and decoding) is in [notebooks/inference.ipynb](notebooks/inference.ipynb).

## Roadmap

- [ ] Sketch-control training
- [ ] More detailed accompaniment training support
- [ ] ONNX export pipeline
- [ ] Interface setup

## Citation

If you use this repo, please cite us at:

```bibtext
@article{novack2026lmdm,
  title         = {Live Music Diffusion Models: Efficient Fine-Tuning and Post-Training of Interactive Diffusion Music Generators},
  author        = {Novack, Zachary and Brade, Stephen and Kim, Haven and Flores Garc{\'i}a, Hugo and Shikarpur, Nithya and Talegaonkar, Chinmay and Kim, Suwan and Chen, Valerie K. and McAuley, Julian and Berg-Kirkpatrick, Taylor and Huang, Cheng-Zhi Anna},
  journal       = {arXiv preprint arXiv:2605.22717},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {2605.22717},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2605.22717}
}
```