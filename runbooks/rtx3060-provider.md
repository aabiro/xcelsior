# RTX 3060 Provider Quickstart

Use the tower RTX 3060 as Xcelsior's low-end provider baseline while larger cards are unavailable.

## 1. Inspect the built-in profile

```bash
python cli.py host-profile 3060
```

The profile models the RTX 3060 as a local-dev tier:

- GPU: RTX 3060
- VRAM: 12 GB total, 11 GB usable for scheduling guidance
- Reference rate: CAD $0.16/hr
- Best use: dev/staging inference, CUDA smoke tests, telemetry burn-in, worker lifecycle testing
- Avoid: 24 GB+ diffusion pipelines, 70B serving, multi-GPU/MIG work

## 2. Run local acceptance

```bash
python cli.py host-accept --host-id tower-server --expected-gpu "RTX 3060" --min-vram 12
```

This checks:

- local Docker, runc, NVIDIA driver, and NVIDIA Container Toolkit versions
- node admission gate results
- NVML/nvidia-smi GPU telemetry
- expected GPU model and minimum VRAM
- recommended runtime

To also verify GPU visibility from inside a container:

```bash
python cli.py host-accept \
  --host-id tower-server \
  --expected-gpu "RTX 3060" \
  --min-vram 12 \
  --docker-probe
```

The Docker probe uses `nvidia/cuda:12.4.1-base-ubuntu22.04`; pre-pull it on the tower if the host should not pull during acceptance.

## 3. Register the tower host

The profile command prints the exact registration command. The default is:

```bash
python cli.py host-add \
  --id tower-server \
  --ip tower-server \
  --gpu "RTX 3060" \
  --vram 12 \
  --free-vram 11 \
  --rate 0.16 \
  --country CA \
  --province ON
```

Adjust `--ip` and `--province` to the actual tailnet name/address and host province.

## 4. Dogfood the scheduler

Use the 3060 for jobs that should fit:

```bash
python cli.py run --model cuda-smoke --vram 2 --gpu "RTX 3060"
python cli.py run --model small-inference --vram 8 --gpu "RTX 3060"
```

Then intentionally submit work that should not fit:

```bash
python cli.py run --model oversized-vram --vram 16 --gpu "RTX 3060"
```

The useful outcome is not raw speed. The useful outcome is proving that admission, matching, telemetry, logs, billing, failed-job handling, and queue diagnostics behave correctly on a real consumer GPU.
