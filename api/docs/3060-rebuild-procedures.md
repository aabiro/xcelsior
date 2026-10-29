# 3060 Tower Rebuild Procedures

**Last Updated**: May 31, 2026 (after reformat)

This document captures the detailed step-by-step procedures to restore the infrastructure state documented in `3060-utilization-plan.md`. These procedures were executed in late April 2026 and should be re-run after any tower reformat/reinstall.

## Prerequisites
- Tower (aaryn@100.64.0.4): base OS fresh, ssh key available, tailscale connected
- VPS (linuxuser@149.28.121.61): ntfy-relay.service and SSH gateway already running
- `/srv/models/` bucket layout exists (restore from backup if needed)

---

## Phase A: ntfy Access Token Wiring (VPS only)

### Context
The Alertmanager → ntfy.sh integration at `https://ntfy.sh/tower-b1gC0ck` used to send account credentials in plaintext. Upgrade to bearer token auth for security.

- **ntfy Token**: `tk_x16f2y5j1lku1wwr7eh3etrhxm6kp` (replace with current token from ntfy.sh account)
- **ntfy Topic**: `tower-b1gC0ck` (must be reserved on https://ntfy.sh/account → Reservations by account owner)
- **VPS Target**: `/usr/local/bin/ntfy-relay.py` (stdlib-only, no requests library)

### Step A1: Patch ntfy-relay.py on VPS

The relay is a simple Python script that forwards Alertmanager webhooks to ntfy.sh. It uses `urllib.request` (stdlib), not `requests`.

**On VPS**:
```bash
ssh -i ~/.ssh/xcelsior linuxuser@149.28.121.61 'sudo bash -c "
# Backup original
cp /usr/local/bin/ntfy-relay.py /usr/local/bin/ntfy-relay.py.bak.\$(date +%s)

# Patch: add NTFY_TOKEN support
python3 <<PATCH
import os
src = open(\"/usr/local/bin/ntfy-relay.py\").read()
if \"NTFY_TOKEN\" in src:
  print(\"Already patched.\")
else:
  # Insert at line 11 (after imports, before post_ntfy function):
  lines = src.split(\"\\n\")
  lines.insert(11, \"NTFY_TOKEN = os.environ.get(\\\"NTFY_TOKEN\\\", \\\"\\\")\")
  src = \"\\n\".join(lines)
  
  # Modify post_ntfy() headers dict (around line 28) to add Authorization
  src = src.replace(
    \"headers = {\",
    \"headers = {\\n        **({\\\\"Authorization\\\\": f\\\"Bearer {NTFY_TOKEN}\\\"} if NTFY_TOKEN else {}),\"
  )
  open(\"/usr/local/bin/ntfy-relay.py\", \"w\").write(src)
  print(\"Patched OK\")
PATCH
"'
```

**Verify**:
```bash
ssh linuxuser@149.28.121.61 'sudo grep -n "NTFY_TOKEN" /usr/local/bin/ntfy-relay.py'
```

Should show:
```
12:NTFY_TOKEN = os.environ.get("NTFY_TOKEN", "")
28:        **({"Authorization": f"Bearer {NTFY_TOKEN}"} if NTFY_TOKEN else {}),
```

### Step A2: Deploy NTFY_TOKEN env file on VPS

```bash
ssh -i ~/.ssh/xcelsior linuxuser@149.28.121.61 'sudo bash -c "
# Create env file (root-owned, readable only by ntfy-relay user)
install -d -m 0750 -o root -g ntfy-relay /etc/ntfy-relay
echo \"NTFY_TOKEN=tk_x16f2y5j1lku1wwr7eh3etrhxm6kp\" | sudo tee /etc/ntfy-relay/env >/dev/null
sudo chmod 0640 /etc/ntfy-relay/env

# Create systemd drop-in to expose env file
sudo mkdir -p /etc/systemd/system/ntfy-relay.service.d
sudo tee /etc/systemd/system/ntfy-relay.service.d/token.conf >/dev/null <<EOF
[Service]
EnvironmentFile=/etc/ntfy-relay/env
EOF

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ntfy-relay

# Verify service is active
sudo systemctl is-active ntfy-relay
"'
```

### Step A3: Smoke test

Send a test Alertmanager alert through the relay:

```bash
ssh linuxuser@149.28.121.61 'curl -sS -m 5 -X POST -H "Content-Type: application/json" \
  -d "{\"alerts\":[{\"status\":\"firing\",\"labels\":{\"alertname\":\"NtfyTokenTest\",\"severity\":\"info\"},\"annotations\":{\"summary\":\"Token wiring verified\",\"description\":\"Bearer auth active\"}}]}" \
  http://127.0.0.1:9099/alert && echo " -> OK"'
```

**Expected**: HTTP 200 response. Check relay logs:
```bash
ssh linuxuser@149.28.121.61 'sudo journalctl -u ntfy-relay -n 3 --no-pager | tail -3'
```

Should show: `"POST /alert HTTP/1.1" 200 -`

### Step A4: Reserve topic on ntfy.sh (manual step)

1. Log in to https://ntfy.sh with the account that owns `tk_x16f2y5j1lku1wwr7eh3etrhxm6kp`
2. Go to **Account** → **Reservations**
3. Click **Reserve Topic** and enter `tower-b1gC0ck`
4. Confirm reservation

This makes the topic private to your account. Without reservation, topics are world-readable.

---

## Phase B: HuggingFace Safetensors Download (Tower only)

### Context
TRT-LLM requires model weights in HuggingFace `safetensors` format, not GGUF blobs. Download Qwen2.5-Coder-32B-Instruct (~62 GB) as the source of truth for conversion.

- **Model**: `Qwen/Qwen2.5-Coder-32B-Instruct` (BF16 format, 14 shards + metadata)
- **Download target**: `/srv/models/hub/huggingface/Qwen/Qwen2.5-Coder-32B-Instruct/`
- **Promotion target**: `/srv/models/lib/llm/qwen2.5-coder-32b-instruct-hf/`
- **Disk required**: 62 GB + 30% headroom for conversion (~80 GB total)

### Step B1: Check disk space on tower

```bash
ssh aaryn@100.64.0.4 'df -h /srv && ls -lah /srv/models/'
```

Verify `/srv` has at least 80 GB free.

### Step B2: Install huggingface-cli on tower

```bash
ssh aaryn@100.64.0.4 'pip3 install --user --break-system-packages "huggingface_hub[cli]" 2>&1 | tail -5'
```

### Step B3: Download in background tmux session

```bash
ssh aaryn@100.64.0.4 '
mkdir -p /srv/models/hub/huggingface/Qwen

cat > /tmp/hf-pull.py <<EOF
from huggingface_hub import snapshot_download
import os
os.makedirs("/srv/models/hub/huggingface/Qwen", exist_ok=True)
print("Starting download …", flush=True)
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    local_dir="/srv/models/hub/huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    local_dir_use_symlinks=False,
    max_workers=4,
)
print("DONE →", path, flush=True)
EOF

tmux new-session -d -s hf-pull "python3 /tmp/hf-pull.py 2>&1 | tee /tmp/hf-pull.log"
echo "Download started in tmux session hf-pull"
tmux ls
'
```

**Monitor progress**:
```bash
ssh aaryn@100.64.0.4 'tail -10 /tmp/hf-pull.log && du -sh /srv/models/hub/huggingface/Qwen/Qwen2.5-Coder-32B-Instruct/'
```

Expected time: 30–60 min depending on HF CDN bandwidth.

### Step B4: Promote to lib/llm bucket (when download finishes)

Once `hf-pull.log` shows "DONE →", hard-link into the canonical location:

```bash
ssh aaryn@100.64.0.4 '
SRC=/srv/models/hub/huggingface/Qwen/Qwen2.5-Coder-32B-Instruct
DST=/srv/models/lib/llm/qwen2.5-coder-32b-instruct-hf

mkdir -p "$DST"

# Hard-link all files (same filesystem, free storage)
for f in "$SRC"/*; do
  [ -f "$f" ] && ln -f "$f" "$DST/$(basename "$f")"
done

chmod -R a+r "$DST"

# Create metadata card
cat > "$DST/CARD.md" <<CARD
# qwen2.5-coder-32b-instruct-hf

- **Source**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- **Format**: HuggingFace safetensors (BF16)
- **Size**: ~62 GB across 14 shards + tokenizer + config
- **Purpose**: TRT-LLM conversion source (int4_awq quantization for sm_86/3060)
- **Post-conversion VRAM targets**:
  - int4_awq: ~16-18 GB (still needs CPU offload on 12GB 3060)
  - q4_k_s GGUF: ~5-6 GB (already available in lib/gguf/)
- **Registered**: $(date -u +%FT%TZ)
CARD

echo "=== Promotion complete ==="
ls -lh "$DST" | head -10
du -sh "$DST"
'
```

---

## Phase C: TRT-LLM Container Setup & Conversion Script (Tower only)

### Context
TRT-LLM is NVIDIA's optimized LLM inference framework. We pull the prebuilt container and stage a conversion script that quantizes the HF safetensors to int4_awq format (compatible with sm_86 GPU, 3060).

- **Container**: `nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3` (~25 GB)
- **Script location**: `/srv/models/_meta/convert-qwen32b-int4awq.sh`
- **Conversion time**: ~30 min on 3060

### Step C1: Troubleshooting note — NIC offload bug

If the `docker pull` fails with `unpigz: corrupted -- crc32 mismatch`, it's likely the r8169 NIC driver bug. **Temporary workaround**:

```bash
ssh aaryn@100.64.0.4 'sudo ethtool -k enp38s0 | grep -E "tx-checksumming|generic-segmentation|generic-receive"'
```

If enabled, temporarily disable (for this pull only):
```bash
ssh aaryn@100.64.0.4 'sudo ethtool -K enp38s0 tx off rx off tso off gso off gro off'
```

After the pull completes, **immediately re-enable**:
```bash
ssh aaryn@100.64.0.4 'sudo ethtool -K enp38s0 tx on rx on tso on gso on gro on'
```

**Document in §7 of plan**: this is a hardware quirk but should be temporary, not persistent.

### Step C2: Pull TRT-LLM image in background

```bash
ssh aaryn@100.64.0.4 '
IMAGE=nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3

tmux new-session -d -s trtpull "sudo docker pull $IMAGE 2>&1 | tee /tmp/trtpull.log"
echo "Pull started in tmux session trtpull"
tmux ls
'
```

**Monitor**:
```bash
ssh aaryn@100.64.0.4 'tail -5 /tmp/trtpull.log; sudo docker images | grep tritonserver'
```

Expected time: 10–30 min depending on CDN.

### Step C3: Stage conversion script

While the image is downloading, create the conversion script:

```bash
ssh aaryn@100.64.0.4 'cat > /srv/models/_meta/convert-qwen32b-int4awq.sh <<"BASH"
#!/usr/bin/env bash
# Convert Qwen2.5-Coder-32B-Instruct (HF safetensors) → TRT-LLM int4_awq engine
# Target: sm_86 (RTX 3060, 12 GB VRAM)
# Execution: run from host; mounts /srv/models into container

set -euo pipefail

IMAGE="nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3"
SRC="/srv/models/lib/llm/qwen2.5-coder-32b-instruct-hf"
WORK="/srv/models/staging/qwen32b-trt-int4awq"
OUT="/srv/models/engines/qwen2.5-coder-32b-instruct-int4awq-sm86"
LOG="/tmp/trt-convert-$(date +%s).log"

mkdir -p "$WORK" "$OUT"

echo "[convert] Step 1/2 — quantize HF → TRT checkpoint (int4_awq)" | tee "$LOG"
sudo docker run --rm --gpus all \
  -v /srv/models:/srv/models \
  -w /opt/tritonserver \
  "$IMAGE" \
  bash -c "
    set -e
    pip install --quiet --no-cache-dir datasets peft
    cd /opt/tritonserver/tensorrt_llm/examples/qwen || cd /opt/tensorrt_llm/examples/qwen
    python3 ../quantization/quantize.py \
      --model_dir $SRC \
      --output_dir $WORK \
      --dtype float16 \
      --qformat int4_awq \
      --awq_block_size 128 \
      --calib_size 32
  " 2>&1 | tee -a "$LOG"

echo "[convert] Step 2/2 — build TRT engine" | tee -a "$LOG"
sudo docker run --rm --gpus all \
  -v /srv/models:/srv/models \
  "$IMAGE" \
  trtllm-build \
    --checkpoint_dir "$WORK" \
    --output_dir "$OUT" \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --max_num_tokens 8192 2>&1 | tee -a "$LOG"

echo "[convert] DONE → $OUT"
echo "Log: $LOG"
ls -lh "$OUT"
BASH
chmod +x /srv/models/_meta/convert-qwen32b-int4awq.sh
echo "Script staged at /srv/models/_meta/convert-qwen32b-int4awq.sh"
'
```

### Step C4: Run conversion (when image pull finishes)

Once the TRT-LLM image is loaded (`sudo docker images | grep tritonserver` shows it):

```bash
ssh aaryn@100.64.0.4 'bash /srv/models/_meta/convert-qwen32b-int4awq.sh'
```

This will:
1. Quantize the 62 GB BF16 model to int4_awq (~8-10 GB intermediate checkpoint)
2. Build the TRT engine (~6-8 GB final binary)
3. Output logs to `/tmp/trt-convert-<timestamp>.log`

**Monitor**:
```bash
ssh aaryn@100.64.0.4 'tail -50 /tmp/trt-convert-*.log | tail -30'
```

---

## Phase D: Documentation Updates (repo only)

Update `docs/3060-utilization-plan.md` with:

### §2.5 Bucket Layout
Add:
```markdown
- **`lib/llm/<id>-hf/`**: HuggingFace safetensors source-of-truth for TRT-LLM conversion
  - Example: `qwen2.5-coder-32b-instruct-hf/` (14 shards + tokenizer + CARD.md)
  - Workflow: download to `hub/huggingface/<name>/`, hard-link to `lib/llm/`, convert to `engines/`
```

### §6 Nightly Backup Timer
✅ Already implemented as `/etc/systemd/system/xcm-prune.timer` (Sunday 04:00 UTC). Verify:
```bash
ssh aaryn@100.64.0.4 'systemctl list-timers xcm-prune.timer'
```

### §7 Hardware Quirks & Decisions
Add:
```markdown
### r8169 NIC Offload Bug

**Hardware**: Realtek RTX r8169 Gigabit NIC (common on B450 motherboards)

**Issue**: High-speed downloads (e.g., Docker image pulls) occasionally corrupt gzip layers with `unpigz: corrupted -- crc32 mismatch`. Root cause is a checksum offload bug in the driver's TX path when combined with TLS streaming.

**Workaround (temporary, per-operation)**:
```bash
# Before large download:
sudo ethtool -K enp38s0 tx off rx off tso off gso off gro off

# After download succeeds:
sudo ethtool -K enp38s0 tx on rx on tso on gso on gro on
```

**Why temporary?** Hardware offload improves throughput for normal traffic. Only disable during problematic operations, then immediately re-enable. Do NOT make persistent (performance impact).

**Alternative (not tested)**: Update r8169 driver or swap NIC for Intel i225 (~$30, no offload bug, slightly higher throughput).
```

---

## Phase E: sm86 ORT TensorRT cache (pixelspark-gpu on tower)

The GHCR `pixelspark-serverless` image ships TensorRT 8 (`libnvinfer.so.8`). Host tower has TensorRT 10/11. ORT 1.18 needs `libnvinfer.so.10` for `TensorrtExecutionProvider`.

**Do not** bind-mount all of `/usr/lib/x86_64-linux-gnu` into the container — `ldconfig` picks up host `libc.so.6` and the container crashes with `GLIBC_PRIVATE` symbol errors.

### Step E1: Copy host TRT libs to a curated directory

```bash
TRT_DIR=/srv/models/engines/sm86/trt_host_libs
rm -rf "$TRT_DIR" && mkdir -p "$TRT_DIR"
for f in /usr/lib/x86_64-linux-gnu/libnvinfer*.so.* /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so.*; do
  [ -f "$f" ] && cp -a "$f" "$TRT_DIR/"
done
cd "$TRT_DIR"
for f in *.so.*.*; do
  base=${f%.so.*}
  major=$(echo "$f" | sed -n 's/.*\.so\.\([0-9]*\).*/\1/p')
  ln -sf "$f" "${base}.so.${major}"
  ln -sf "$f" "${base}.so"
done
```

Expect ~6 GB under `trt_host_libs/`.

### Step E2: Mount in `docker-compose.tower.yml`

```yaml
      - /srv/models/engines/sm86/trt_host_libs:/host-trt-libs:ro
```

Keep `cuda.conf` to CUDA paths only (no `/host-trt-libs` in `ld.so.conf.d`).

### Step E3: Run overnight cache warm

```bash
cd /srv/models/onnx/production/serverless_bundle
nohup ./build_trt_3060_ort.sh &
tail -f /srv/models/engines/sm86/ort_build.log
```

The wrapper sets `LD_LIBRARY_PATH=/host-trt-libs:...` on `docker exec`. Engines land in `/srv/models/engines/sm86/ort_cache/`.

### Step E4: Verify TRT EP before leaving it overnight

```bash
docker exec -e LD_LIBRARY_PATH=/host-trt-libs pixelspark-gpu python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('/models/exposure/1xExposureCorrection_compact_fp32_opset17.onnx',
  providers=[('TensorrtExecutionProvider', {'trt_engine_cache_path': '/engines/sm86/ort_cache'}), 'CUDAExecutionProvider'])
print(sess.get_providers())
"
```

First provider must be `TensorrtExecutionProvider`.

---

## Reference: After-Reformat Checklist

Use this checklist to verify full restoration:

- [ ] **Phase A**: ntfy token wiring
  - [ ] A1: ntfy-relay.py patched on VPS (grep shows NTFY_TOKEN + Authorization)
  - [ ] A2: /etc/ntfy-relay/env deployed, systemd drop-in active
  - [ ] A3: Smoke test passes (200 OK to ntfy.sh, relay logs show Bearer auth)
  - [ ] A4: Topic `tower-b1gC0ck` reserved on https://ntfy.sh/account

- [ ] **Phase B**: HF safetensors download
  - [ ] B1: /srv has ≥80 GB free
  - [ ] B2: huggingface_hub installed on tower
  - [ ] B3: Download in `hf-pull` tmux session completes (~60 min)
  - [ ] B4: Hard-linked to `/srv/models/lib/llm/qwen2.5-coder-32b-instruct-hf/` + CARD.md written

- [ ] **Phase C**: TRT-LLM container & conversion
  - [ ] C1: Docker image pulled to tower (`nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3`)
  - [ ] C2: Conversion script staged at `/srv/models/_meta/convert-qwen32b-int4awq.sh`
  - [ ] C3: Conversion runs and outputs to `/srv/models/engines/qwen2.5-coder-32b-instruct-int4awq-sm86/`

- [ ] **Phase D**: Documentation
  - [ ] D1: §2.5 updated with `lib/llm/<id>-hf/` bucket pattern
  - [ ] D2: §7 documents r8169 offload workaround

---

## Rollback / Revert

If any phase fails, simple reverts:

- **Phase A**: Delete `/etc/ntfy-relay/` and `/etc/systemd/system/ntfy-relay.service.d/`, restore from `.bak` file
- **Phase B**: Delete `/srv/models/hub/huggingface/Qwen/` and `/srv/models/lib/llm/qwen2.5-coder-32b-instruct-hf/`
- **Phase C**: Delete `/srv/models/staging/qwen32b-trt-int4awq/` and `/srv/models/engines/qwen2.5-coder-32b-instruct-int4awq-sm86/`

All are non-destructive; underlying `/srv` bucket persists.

---

## Questions / Debugging

- **NIC corruption persists?** Try `skopeo copy docker://nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 docker-daemon:trt-llm:latest` (different protocol, often more resilient)
- **Disk space?** Check `/srv/models/staging/` — old TRT checkpoints can be deleted after engine is built
- **Conversion times slow?** GPU offload behavior is normal for quantization on limited VRAM; use `--device cpu` mode if GPU runs out of memory
