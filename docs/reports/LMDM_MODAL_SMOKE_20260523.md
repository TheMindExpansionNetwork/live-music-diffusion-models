# LMDM Modal Smoke — 2026-05-23

- App: `sonic-forage-lmdm-smoke`
- Repo: https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git
- Upstream: https://github.com/ZacharyNovack/live-music-diffusion-models.git
- Smoke type: non-generative clone/compile/config/inference-symbol smoke
- Result: `OK`

## Steps

- `git clone --depth 1 https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git /tmp/live-music-diffusion-models` → exit `0`, 0.339s
- `git log -1 --format='%H%n%ci%n%s'` → exit `0`, 0.007s
- `python -m py_compile train.py pre_encode.py run_gradio.py stable_audio_tools/inference/generation.py` → exit `0`, 0.087s
- `python - <<'PY'
import json, pathlib
base = pathlib.Path('stable_audio_tools/configs/model_configs/txt2audio')
configs = sorted(p.name for p in base.glob('*.json'))
for p in base.glob('*.json'):
    json.load(open(p))
print(json.dumps({'config_count': len(configs), 'configs': configs}, indent=2))
PY` → exit `0`, 0.052s
- `python - <<'PY'
from pathlib import Path
text = Path('stable_audio_tools/inference/generation.py').read_text()
needles = ['generate_diffusion_cond_blockar', 'use_kv_cache', 'context_router_attention_pattern']
print({n: (n in text) for n in needles})
assert 'generate_diffusion_cond_blockar' in text
PY` → exit `0`, 0.046s

## Notes

This is intentionally non-generative: it verifies the fresh LMDM fork clones inside Modal, key files compile, text-to-audio configs parse, and the block-autoregressive inference symbol is present. GPU generation/training requires a compatible checkpoint and a separate bounded run.
