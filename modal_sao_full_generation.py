"""Modal GPU full-generation proof for the LMDM fork via Stable Audio Open.

This is a bounded full inference run, not training: it uses the LMDM fork's
stable_audio_tools code path with a compatible Stable Audio Open checkpoint to
render real audio samples in Modal GPU. It also renders an init-audio variation
from the local didgeridoo earth-drone sample.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "lmdm-sao-full-generation"
FORK_URL = "https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git"
MODEL_ID = "stabilityai/stable-audio-open-1.0"
LOCAL_SAMPLE = "/opt/data/workspace/live-music-diffusion/earth-sample/didgeridoo-earth-drone-v2-sample-44k-stereo.wav"
REMOTE_SAMPLE = "/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav"
HF_CACHE = "/cache/huggingface"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("lmdm-sao-hf-cache", create_if_missing=True)

# Keep this heavy GPU image separate from the CPU smoke image so normal smoke runs
# do not accidentally build CUDA/checkpoint layers.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install(
        "numpy==1.23.5",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        "safetensors==0.7.0",
        "huggingface_hub==0.24.7",
        "transformers==4.46.3",
        "sentencepiece==0.1.99",
        "einops",
        "einops-exts",
        "k-diffusion==0.1.1",
        "local-attention==1.8.6",
        "alias-free-torch==0.0.6",
        "auraloss==0.4.0",
        "descript-audio-codec==1.0.0",
        "ema-pytorch==0.2.3",
        "encodec==0.1.1",
        "importlib-resources==5.12.0",
        "laion-clap==1.1.4",
        "pandas==2.0.2",
        "prefigure==0.0.9",
        "pytorch_lightning==2.1.0",
        "PyWavelets==1.4.1",
        "torchmetrics==0.11.4",
        "tqdm",
        "v-diffusion-pytorch==0.0.2",
        "vector-quantize-pytorch==1.14.41",
        "wandb==0.15.4",
        "webdataset==0.2.100",
    )
    .run_commands(
        f"git clone --depth 1 {FORK_URL} /root/live-music-diffusion-models",
        "python -m pip install -e /root/live-music-diffusion-models --no-deps",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_CACHE": f"{HF_CACHE}/hub",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file(LOCAL_SAMPLE, REMOTE_SAMPLE)
)


def _run(cmd: str, cwd: str | None = None, timeout: int = 600) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True, timeout=timeout)
    return {
        "cmd": cmd,
        "cwd": cwd,
        "returncode": p.returncode,
        "seconds": round(time.time() - t0, 3),
        "stdout_tail": p.stdout[-5000:],
        "stderr_tail": p.stderr[-5000:],
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={HF_CACHE: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_generation(seconds_total: int = 8, steps: int = 8) -> dict:
    """Run two bounded real generations and return audio bytes + receipt."""
    work = Path("/root/live-music-diffusion-models")
    out_dir = Path("/tmp/lmdm-sao-generation")
    out_dir.mkdir(parents=True, exist_ok=True)

    code = r'''
import hashlib, json, os, time
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-audio-open-1.0")
SECONDS_TOTAL = int(os.environ.get("SECONDS_TOTAL", "8"))
STEPS = int(os.environ.get("STEPS", "8"))
REMOTE_SAMPLE = Path(os.environ.get("REMOTE_SAMPLE", "/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav"))
OUT = Path(os.environ.get("OUT", "/tmp/lmdm-sao-generation"))
OUT.mkdir(parents=True, exist_ok=True)

def summarize_wav(path: Path):
    wav, sr = torchaudio.load(str(path))
    mono = wav.mean(dim=0)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "sample_rate": sr,
        "channels": int(wav.shape[0]),
        "frames": int(wav.shape[1]),
        "seconds": round(float(wav.shape[1] / sr), 3),
        "rms": round(float(torch.sqrt((wav ** 2).mean())), 6),
        "peak": round(float(wav.abs().max()), 6),
        "mono_first10": [round(float(x), 6) for x in mono[:10]],
    }

def save_audio(name: str, audio: torch.Tensor, sample_rate: int):
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32)
    audio = audio / audio.abs().max().clamp(min=1e-8)
    audio = audio.clamp(-1, 1).cpu()
    path = OUT / f"{name}.wav"
    torchaudio.save(str(path), audio, sample_rate)
    return path

started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside Modal GPU function")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"
model, model_config = get_pretrained_model(MODEL_ID)
model = model.to(device).eval().requires_grad_(False)
# float16 cuts memory enough for the 40GB A100 lane while matching normal inference usage.
model = model.to(torch.float16)
sample_rate = int(model_config["sample_rate"])
sample_size = int(SECONDS_TOTAL * sample_rate)

common = dict(
    model=model,
    steps=STEPS,
    cfg_scale=7.0,
    sample_size=sample_size,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.3,
    sigma_max=500,
    rho=1.0,
    device=device,
)

prompt_drone = "deep didgeridoo earth drone, ritual bass resonance, organic throat harmonics, slow evolving dark ambient texture, no drums"
conditioning_drone = [{"prompt": prompt_drone, "seconds_start": 0, "seconds_total": SECONDS_TOTAL}]
# Avoid negative_conditioning here: this LMDM fork's transformer path does not
# accept the older Stable Audio Open `negative_prepend_cond` keyword.
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
    pure = generate_diffusion_cond(
        **common,
        conditioning=conditioning_drone,
        seed=424242,
    )
pure_path = save_audio("sao_open_earth_drone_prompt", pure, sample_rate)
del pure

earth_audio, earth_sr = torchaudio.load(str(REMOTE_SAMPLE))
# Variation/continuation lane: keep some source identity but let the checkpoint re-synthesize.
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
    varied = generate_diffusion_cond(
        **common,
        conditioning=conditioning_drone,
        init_audio=(earth_sr, earth_audio),
        init_noise_level=0.55,
        seed=424243,
    )
var_path = save_audio("sao_open_earth_drone_init_variation", varied, sample_rate)
del varied

torch.cuda.empty_cache()
receipt = {
    "ok": True,
    "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_id": MODEL_ID,
    "seconds_total_requested": SECONDS_TOTAL,
    "steps": STEPS,
    "sample_rate": sample_rate,
    "model_sample_size": model_config.get("sample_size"),
    "diffusion_objective": getattr(model, "diffusion_objective", None),
    "source_sample": summarize_wav(REMOTE_SAMPLE),
    "outputs": [summarize_wav(pure_path), summarize_wav(var_path)],
    "runtime_seconds": round(time.time() - started, 3),
    "gpu": torch.cuda.get_device_name(0),
    "cuda_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "prompt": prompt_drone,
    "scope": "Full bounded GPU inference/generation through the fork's stable_audio_tools code. Not training and not LMDM ARC checkpoint generation because public LMDM checkpoints are not shipped upstream.",
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
print(json.dumps(receipt, indent=2, sort_keys=True))
'''
    runner = out_dir / "run_sao_generation.py"
    runner.write_text(code)
    env = os.environ.copy()
    env.update({
        "MODEL_ID": MODEL_ID,
        "SECONDS_TOTAL": str(seconds_total),
        "STEPS": str(steps),
        "REMOTE_SAMPLE": REMOTE_SAMPLE,
        "OUT": str(out_dir),
    })
    t0 = time.time()
    p = subprocess.run(
        "python run_sao_generation.py",
        cwd=str(out_dir),
        env=env,
        shell=True,
        text=True,
        capture_output=True,
        timeout=3300,
    )
    step = {
        "cmd": "python run_sao_generation.py",
        "returncode": p.returncode,
        "seconds": round(time.time() - t0, 3),
        "stdout_tail": p.stdout[-8000:],
        "stderr_tail": p.stderr[-8000:],
    }
    if p.returncode != 0:
        return {
            "ok": False,
            "app": APP_NAME,
            "model_id": MODEL_ID,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "step": step,
        }

    receipt = json.loads((out_dir / "receipt.json").read_text())
    files = {}
    for wav in out_dir.glob("*.wav"):
        files[wav.name] = base64.b64encode(wav.read_bytes()).decode("ascii")
    return {
        "ok": True,
        "app": APP_NAME,
        "fork_url": FORK_URL,
        "model_id": MODEL_ID,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt": receipt,
        "step": step,
        "files_b64": files,
    }


@app.local_entrypoint()
def main(out_dir: str = "/opt/data/workspace/live-music-diffusion/modal-sao-full-generation", seconds_total: int = 8, steps: int = 8):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_generation.remote(seconds_total=seconds_total, steps=steps)
    files = result.pop("files_b64", {})
    for name, data in files.items():
        (out / name).write_bytes(base64.b64decode(data))
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    receipt = result.get("receipt") or {}
    report = [
        "# LMDM fork — Stable Audio Open full Modal generation\n",
        f"- App: `{APP_NAME}`",
        f"- Fork: {FORK_URL}",
        f"- Model: `{MODEL_ID}`",
        f"- OK: `{result.get('ok')}`",
        f"- Checked: `{result.get('checked_at_utc')}`",
        f"- Seconds requested: `{seconds_total}`",
        f"- Steps: `{steps}`",
        "",
        "## Scope",
        receipt.get("scope", ""),
        "",
        "## Outputs",
    ]
    for item in receipt.get("outputs", []):
        report += [
            f"- `{Path(item['path']).name}`",
            f"  - seconds: `{item['seconds']}`",
            f"  - bytes: `{item['bytes']}`",
            f"  - sha256: `{item['sha256']}`",
            f"  - rms/peak: `{item['rms']}` / `{item['peak']}`",
        ]
    report += ["", "## Receipt JSON", "```json", json.dumps(receipt, indent=2, sort_keys=True), "```", ""]
    (out / "README_MODAL_SAO_FULL_GENERATION.md").write_text("\n".join(report))
    print(json.dumps(result, indent=2, sort_keys=True))
