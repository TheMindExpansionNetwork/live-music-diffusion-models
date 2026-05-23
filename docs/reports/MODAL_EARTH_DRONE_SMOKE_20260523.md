# Live Music Diffusion Models — Modal earth-drone smoke

- App: `lmdm-earth-drone-smoke`
- Modal run: https://modal.com/apps/m1ndb0t-2045/main/ap-TgbfPpJLWvBVv6nvOTzfz4
- Fork: https://github.com/TheMindExpansionNetwork/live-music-diffusion-models.git
- Upstream: https://github.com/ZacharyNovack/live-music-diffusion-models.git
- Commit tested: `ab7434662ae0522fa1c88062414f90ebfe4f5e44`
- OK: `True`
- Lane: CPU source/preprocess smoke with earth-drone sample; no checkpoint generation
- Sample handling: sample baked into Modal image from local operator path; not committed to public fork

## Smoke JSON

```json
{
  "chunk_shape": [
    2,
    176400
  ],
  "envelope64_first8": [
    0.078721,
    0.064434,
    0.067392,
    0.080391,
    0.078329,
    0.083276,
    0.071451,
    0.075145
  ],
  "input_channels": 2,
  "input_frames": 264600,
  "input_seconds": 6.0,
  "input_sr": 44100,
  "padding_ratio": 1.0,
  "peak": 0.22796630859375,
  "prepared_shape": [
    1,
    2,
    176400
  ],
  "rms": 0.07302537560462952,
  "sample_bytes": 1058478,
  "sample_path": "/root/earth-drone/didgeridoo-earth-drone-v2-sample-44k-stereo.wav",
  "seconds_total": 4
}
```

## Scope

This is a setup/preprocessing smoke. It does not claim full LMDM music generation because upstream checkpoint weights and GPU generation wiring were not provided.

## Local sample source

- Original OGG: `/opt/data/workspace/datasets/didgeridoo-earth-drone-v2/processed/samples/didgeridoo-earth-drone-v2-sample.ogg`
- Converted WAV used for Modal mount: `/opt/data/workspace/live-music-diffusion/earth-sample/didgeridoo-earth-drone-v2-sample-44k-stereo.wav`
- The audio sample is not committed to this public fork.
