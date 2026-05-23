"""Modal smoke test for Live Music Diffusion Models using the local earth-drone sample.

This is intentionally a bounded CPU setup test, not a full checkpoint generation run.
It proves the fresh LMDM fork can be cloned, installed minimally, compile/import its
audio preprocessing path, and process our didgeridoo earth-drone sample inside Modal.

The sample is mounted/baked from the operator machine at run time and is not committed
into the public fork.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "lmdm-earth-drone-smoke"
FORK_URL = "https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git"
UPSTREAM_URL = "https://github.com/ZacharyNovack/live-music-diffusion-models.git"
UPSTREAM_COMMIT = "ab7434662ae0522fa1c88062414f90ebfe4f5e44"
LOCAL_SAMPLE = "/opt/data/workspace/live-music-diffusion/earth-sample/didgeridoo-earth-drone-v2-sample-44k-stereo.wav"
REMOTE_SAMPLE = "/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "safetensors==0.7.0",
        "packaging==26.2",
        "huggingface_hub==1.16.1",
    )
    .add_local_file(LOCAL_SAMPLE, REMOTE_SAMPLE)
)


def _run(cmd: str, cwd: str | None = None, timeout: int = 240) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True, timeout=timeout)
    return {
        "cmd": cmd,
        "cwd": cwd,
        "returncode": p.returncode,
        "seconds": round(time.time() - t0, 3),
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-4000:],
    }


@app.function(image=image, timeout=1200, cpu=2, memory=8192)
def run_smoke() -> dict:
    steps: list[dict] = []
    work = Path("/tmp/live-music-diffusion-models")
    sample = Path(REMOTE_SAMPLE)

    steps.append(_run(f"git clone --depth 1 {FORK_URL} {work}", timeout=180))
    steps.append(_run(f"git fetch --depth 1 origin {UPSTREAM_COMMIT}", cwd=str(work), timeout=120))
    steps.append(_run(f"git checkout {UPSTREAM_COMMIT}", cwd=str(work), timeout=120))
    actual_commit = subprocess.check_output("git rev-parse HEAD", cwd=work, shell=True, text=True).strip()

    # Minimal install: the upstream package's full dependency list pulls many heavy training deps.
    # This smoke only needs the source tree plus CPU torch/torchaudio/numpy for preprocessing.
    steps.append(_run("python -m pip install -e . --no-deps", cwd=str(work), timeout=240))
    steps.append(_run("python -m compileall -q stable_audio_tools/inference stable_audio_tools/data stable_audio_tools/models", cwd=str(work), timeout=180))

    smoke_code = r'''
from pathlib import Path
import json, math, wave
import numpy as np
import torch
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.data.utils import PadCrop_Normalized_T

sample = Path("/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav")
with wave.open(str(sample), "rb") as wf:
    sr = wf.getframerate()
    ch = wf.getnchannels()
    n = wf.getnframes()
    width = wf.getsampwidth()
    raw = wf.readframes(n)
if width != 2:
    raise RuntimeError(f"expected int16 wav sample, got sampwidth={width}")
arr = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
arr = arr.reshape(-1, ch).T
audio = torch.from_numpy(arr)
prepared = prepare_audio(audio, in_sr=sr, target_sr=44100, target_length=44100 * 4, target_channels=2, device="cpu")
chunk, t_start, t_end, seconds_start, seconds_total, padding_mask = PadCrop_Normalized_T(44100 * 4, 44100, randomize=False)(prepared[0])
# A tiny deterministic derived preview: 64 RMS envelope values over the prepared block.
mono = prepared[0].mean(dim=0)[: 44100 * 4]
bounds = torch.linspace(0, mono.numel(), 65, dtype=torch.long)
envelope = []
for i in range(64):
    seg = mono[bounds[i].item():bounds[i + 1].item()]
    envelope.append(float(torch.sqrt((seg ** 2).mean())))
print(json.dumps({
    "sample_path": str(sample),
    "sample_bytes": sample.stat().st_size,
    "input_sr": sr,
    "input_channels": ch,
    "input_frames": n,
    "input_seconds": n / sr,
    "prepared_shape": list(prepared.shape),
    "chunk_shape": list(chunk.shape),
    "padding_ratio": float(padding_mask.mean()),
    "seconds_total": int(seconds_total),
    "rms": float(torch.sqrt((prepared ** 2).mean())),
    "peak": float(prepared.abs().max()),
    "envelope64_first8": [round(float(x), 6) for x in envelope[:8]],
}, sort_keys=True))
'''
    (work / "earth_drone_preprocess_smoke.py").write_text(smoke_code)
    smoke_step = _run("python earth_drone_preprocess_smoke.py", cwd=str(work), timeout=180)
    steps.append(smoke_step)

    smoke = None
    if smoke_step["returncode"] == 0:
        smoke = json.loads(smoke_step["stdout_tail"].strip().splitlines()[-1])

    ok = all(s["returncode"] == 0 for s in steps) and bool(smoke)
    return {
        "ok": ok,
        "app": APP_NAME,
        "fork_url": FORK_URL,
        "upstream_url": UPSTREAM_URL,
        "commit": actual_commit,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "lane": "CPU source/preprocess smoke with earth-drone sample; no checkpoint generation",
        "sample_note": "sample baked into Modal image from local operator path; not committed to public fork",
        "smoke": smoke,
        "steps": steps,
    }


@app.local_entrypoint()
def main(out_dir: str = "/opt/data/workspace/live-music-diffusion/modal-earth-drone-receipt"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_smoke.remote()
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "README_MODAL_EARTH_DRONE_SMOKE.md").write_text(
        "# Live Music Diffusion Models — Modal earth-drone smoke\n\n"
        f"- App: `{APP_NAME}`\n"
        f"- Fork: {FORK_URL}\n"
        f"- Upstream: {UPSTREAM_URL}\n"
        f"- Commit tested: `{result.get('commit')}`\n"
        f"- OK: `{result.get('ok')}`\n"
        f"- Lane: {result.get('lane')}\n"
        f"- Sample handling: {result.get('sample_note')}\n\n"
        "## Smoke JSON\n\n```json\n"
        + json.dumps(result.get("smoke"), indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Scope\n\n"
        "This is a setup/preprocessing smoke. It does not claim full LMDM music generation because upstream checkpoint weights and GPU generation wiring were not provided.\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
