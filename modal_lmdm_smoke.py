import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "sonic-forage-lmdm-smoke"
REPO_URL = "https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git"
UPSTREAM_URL = "https://github.com/ZacharyNovack/live-music-diffusion-models.git"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ca-certificates")
)


def run(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
    start = time.time()
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "cwd": cwd,
        "exit_code": proc.returncode,
        "seconds": round(time.time() - start, 3),
        "output_tail": proc.stdout[-6000:],
    }


@app.function(image=image, timeout=1200)
def smoke() -> dict:
    root = Path("/tmp/live-music-diffusion-models")
    steps: list[dict] = []
    steps.append(run(f"git clone --depth 1 {REPO_URL} {root}"))
    steps.append(run("git log -1 --format='%H%n%ci%n%s'", cwd=str(root)))
    steps.append(run("python -m py_compile train.py pre_encode.py run_gradio.py stable_audio_tools/inference/generation.py", cwd=str(root)))
    steps.append(run("python - <<'PY'\nimport json, pathlib\nbase = pathlib.Path('stable_audio_tools/configs/model_configs/txt2audio')\nconfigs = sorted(p.name for p in base.glob('*.json'))\nfor p in base.glob('*.json'):\n    json.load(open(p))\nprint(json.dumps({'config_count': len(configs), 'configs': configs}, indent=2))\nPY", cwd=str(root)))
    steps.append(run("python - <<'PY'\nfrom pathlib import Path\ntext = Path('stable_audio_tools/inference/generation.py').read_text()\nneedles = ['generate_diffusion_cond_blockar', 'use_kv_cache', 'context_router_attention_pattern']\nprint({n: (n in text) for n in needles})\nassert 'generate_diffusion_cond_blockar' in text\nPY", cwd=str(root)))
    ok = all(s["exit_code"] == 0 for s in steps)
    return {
        "ok": ok,
        "app": APP_NAME,
        "repo_url": REPO_URL,
        "upstream_url": UPSTREAM_URL,
        "smoke_type": "non-generative clone/compile/config/inference-symbol smoke",
        "steps": steps,
    }


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print(json.dumps(result, indent=2))
    out_dir = Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lmdm_modal_smoke_receipt_20260523.json"
    out.write_text(json.dumps(result, indent=2))
    md = out_dir / "LMDM_MODAL_SMOKE_20260523.md"
    md.write_text(
        "# LMDM Modal Smoke — 2026-05-23\n\n"
        f"- App: `{APP_NAME}`\n"
        f"- Repo: {REPO_URL}\n"
        f"- Upstream: {UPSTREAM_URL}\n"
        f"- Smoke type: {result['smoke_type']}\n"
        f"- Result: `{'OK' if result['ok'] else 'FAIL'}`\n\n"
        "## Steps\n\n"
        + "\n".join(
            f"- `{s['cmd']}` → exit `{s['exit_code']}`, {s['seconds']}s" for s in result["steps"]
        )
        + "\n\n## Notes\n\n"
        "This is intentionally non-generative: it verifies the fresh LMDM fork clones inside Modal, key files compile, text-to-audio configs parse, and the block-autoregressive inference symbol is present. GPU generation/training requires a compatible checkpoint and a separate bounded run.\n"
    )
    if not result.get("ok"):
        raise SystemExit(1)
