# LMDM fork — Stable Audio Open full Modal generation

- App: `lmdm-sao-full-generation`
- Modal run: https://modal.com/apps/m1ndb0t-2045/main/ap-SHeigMmNIWznKJcT1PP00L
- Fork: https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git
- Model: `stabilityai/stable-audio-open-1.0`
- OK: `True`
- Checked: `2026-05-23T14:16:13.003855+00:00`
- Seconds requested: `8`
- Steps: `8`

## Scope
Full bounded GPU inference/generation through the fork's stable_audio_tools code. Not training and not LMDM ARC checkpoint generation because public LMDM checkpoints are not shipped upstream.

## Outputs
- `sao_open_earth_drone_prompt.wav`
  - seconds: `7.988`
  - bytes: `1409102`
  - sha256: `3ddfa7897fa167543a435381e9db9a7a48d4707ea7c081a38ab26e84bf3da155`
  - rms/peak: `0.318366` / `0.999969`
- `sao_open_earth_drone_init_variation.wav`
  - seconds: `7.988`
  - bytes: `1409102`
  - sha256: `349e4707f8dee5fcabf9cbff2a94ddb90b3d7b00997608827a6e5295ad758c7f`
  - rms/peak: `0.159073` / `0.999969`

## Receipt JSON
```json
{
  "checked_at_utc": "2026-05-23T14:16:11Z",
  "cuda_mem_allocated_gb": 4.524,
  "diffusion_objective": "v",
  "gpu": "NVIDIA A100-SXM4-40GB",
  "model_id": "stabilityai/stable-audio-open-1.0",
  "model_sample_size": 2097152,
  "ok": true,
  "outputs": [
    {
      "bytes": 1409102,
      "channels": 2,
      "frames": 352256,
      "mono_first10": [
        0.158432,
        0.296646,
        0.227158,
        0.383194,
        0.224197,
        0.393814,
        0.177078,
        0.325333,
        0.173355,
        0.222321
      ],
      "path": "/tmp/lmdm-sao-generation/sao_open_earth_drone_prompt.wav",
      "peak": 0.999969,
      "rms": 0.318366,
      "sample_rate": 44100,
      "seconds": 7.988,
      "sha256": "3ddfa7897fa167543a435381e9db9a7a48d4707ea7c081a38ab26e84bf3da155"
    },
    {
      "bytes": 1409102,
      "channels": 2,
      "frames": 352256,
      "mono_first10": [
        0.014069,
        0.017166,
        0.016052,
        0.020142,
        0.020569,
        0.025787,
        0.027466,
        0.031662,
        0.03244,
        0.034622
      ],
      "path": "/tmp/lmdm-sao-generation/sao_open_earth_drone_init_variation.wav",
      "peak": 0.999969,
      "rms": 0.159073,
      "sample_rate": 44100,
      "seconds": 7.988,
      "sha256": "349e4707f8dee5fcabf9cbff2a94ddb90b3d7b00997608827a6e5295ad758c7f"
    }
  ],
  "prompt": "deep didgeridoo earth drone, ritual bass resonance, organic throat harmonics, slow evolving dark ambient texture, no drums",
  "runtime_seconds": 17.823,
  "sample_rate": 44100,
  "scope": "Full bounded GPU inference/generation through the fork's stable_audio_tools code. Not training and not LMDM ARC checkpoint generation because public LMDM checkpoints are not shipped upstream.",
  "seconds_total_requested": 8,
  "source_sample": {
    "bytes": 1058478,
    "channels": 2,
    "frames": 264600,
    "mono_first10": [
      0.021667,
      0.023621,
      0.021881,
      0.023041,
      0.022461,
      0.023346,
      0.023041,
      0.023956,
      0.02417,
      0.024872
    ],
    "path": "/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav",
    "peak": 0.324402,
    "rms": 0.071343,
    "sample_rate": 44100,
    "seconds": 6.0,
    "sha256": "52000bab98882c7733930f795fa40facf731f1eb0461daa2f0912d74b9529fa1"
  },
  "steps": 8
}
```

## Local sample delivery

Generated audio samples were delivered locally under `/opt/data/drops/lmdm-sao-full-generation/` and are not committed to this public fork.
