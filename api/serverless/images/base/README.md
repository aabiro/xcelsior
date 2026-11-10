# Xcelsior Serverless Base (`ghcr.io/xcelsior-gpu/serverless-base`)

Official base image and Python Worker SDK (`serverless.worker_sdk`) for custom queue and ASGI workers on [Xcelsior Serverless](https://xcelsior.ca).

```dockerfile
FROM ghcr.io/xcelsior-gpu/serverless-base:py3.12
```

- **Registry:** `ghcr.io/xcelsior-gpu/serverless-base`
- **Tags:**
  - `py3.12` — tracks the latest verified release on `main`
  - `py3.12-<commit>` — immutable 12-character commit tag
- **Base OS:** `python:3.12-slim-bookworm` (`linux/amd64`)
- **Docs:** <https://docs.xcelsior.ca>

---

## What Is Included

| Component | Details |
|---|---|
| `serverless.worker_sdk` | Installed at `/opt/xcelsior/serverless/worker_sdk` (`PYTHONPATH=/opt/xcelsior`) |
| Runtime dependencies | `httpx==0.28.1`, `uvicorn==0.40.0`, `fastapi==0.135.4`, `curl`, `ca-certificates` |
| Health check | Built-in `HEALTHCHECK` probing `http://127.0.0.1:${XCELSIOR_HTTP_PORT}/healthz` (default port `8080`) |
| Model cache path | `HF_HOME=/xcelsior-cache/huggingface` |

There is no CUDA toolkit baked into this base image. On Xcelsior GPU hosts, the NVIDIA container runtime injects the host driver automatically, and Python ML wheels (such as `torch`, `vllm`, `onnxruntime-gpu`) bundle their own CUDA runtime libraries. See [Using Full CUDA Toolkit (`nvcc`)](#using-full-cuda-toolkit-nvcc) below if your worker compiles custom CUDA kernels at build or run time.

---

## Quick Start: Queue-Mode Worker

Queue-mode workers claim jobs from the Xcelsior control plane, run your Python function, report token/compute usage and streaming events, and drain cleanly on `SIGTERM`.

### 1. Write `handler.py`

```python
from serverless.worker_sdk import handler, run_worker


@handler
def process(job: dict) -> dict:
    payload = job.get("input") or {}
    prompt = payload.get("prompt", "")
    return {
        "output": {"text": f"Processed: {prompt}"},
        "usage": {
            "input_tokens": 12,
            "output_tokens": 24,
        },
    }


if __name__ == "__main__":
    run_worker()
```

Your `@handler` function can be synchronous (`def`) or asynchronous (`async def`). It can return:
- A plain `dict` (sent directly as the job's `output`), or
- A `dict` containing `"output"` and `"usage"` (`input_tokens`, `output_tokens`, `cached_tokens`, `ttft_ms`).

### 2. Write `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM ghcr.io/xcelsior-gpu/serverless-base:py3.12

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
CMD ["python", "handler.py"]
```

### 3. Smoke-Test Locally

```bash
docker build -t my-worker:latest .
docker run --rm -p 8080:8080 \
  -e XCELSIOR_SKIP_CUDA_CHECK=1 \
  -e XCELSIOR_WORKER_ID=local-smoke \
  my-worker:latest
```

Verify readiness in another terminal:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"ok": true, "draining": false}
```

---

## Streaming Events & Error Reporting

Emit progress updates, log lines, or incremental output chunks while a job is running:

```python
from serverless.worker_sdk import (
    WorkerError,
    handler,
    log_event,
    output_chunk,
    progress_event,
    run_worker,
)


@handler
def process(job: dict) -> dict:
    prompt = (job.get("input") or {}).get("prompt")
    if not prompt:
        raise WorkerError("Missing required 'prompt' field in input", code="invalid_input")

    log_event("Starting inference")
    progress_event(0.5, message="Halfway through generation")
    output_chunk({"delta": "Partial token stream..."})

    return {"result": "done"}


if __name__ == "__main__":
    run_worker()
```

---

## ASGI / OpenAI-Compatible HTTP Worker Mode

If your custom container serves an HTTP/ASGI application (such as a custom FastAPI server or OpenAI-compatible `/v1/chat/completions` proxy), use `serve_asgi`:

```python
from fastapi import FastAPI
from serverless.worker_sdk import serve_asgi

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    return {"choices": [{"message": {"role": "assistant", "content": "Hello from Xcelsior"}}]}


if __name__ == "__main__":
    serve_asgi(app)
```

---

## Environment Variables & Boot Fitness Checks

On startup, `run_worker()` runs `run_fitness_checks()` before opening `/healthz`:

| Variable | Default | Description |
|---|---|---|
| `XCELSIOR_HTTP_PORT` | `8080` | Port for the built-in `/healthz` server (or `serve_asgi`) |
| `XCELSIOR_MAX_CONCURRENCY` | `1` | Number of jobs claimed and executed concurrently per worker |
| `XCELSIOR_SKIP_CUDA_CHECK` | `0` | Set to `1` for CPU-only local testing without an NVIDIA GPU |
| `XCELSIOR_WEIGHTS_READY_PATH` | *(unset)* | If set, boot fails unless this file/directory exists before serving |
| `XCELSIOR_MODEL_CACHE_MOUNT` | *(unset)* | Additional mount path verified for minimum free disk space (`5 GB` default) |
| `HF_HOME` | `/xcelsior-cache/huggingface` | Default Hugging Face cache directory |

When the host scales down or rotates a worker, it sends `SIGTERM`. The worker immediately flips `/healthz` to `{"ok": false, "draining": true}`, stops claiming new jobs, waits for in-flight jobs to finish, and exits `0`.

---

## Using Full CUDA Toolkit (`nvcc`)

If your worker needs the full NVIDIA CUDA toolkit (`nvcc`, headers, static libs) rather than wheel-bundled CUDA runtime libraries, start from `nvidia/cuda` and copy `/opt/xcelsior/serverless` from this image:

```dockerfile
FROM ghcr.io/xcelsior-gpu/serverless-base:py3.12 AS sdk
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

COPY --from=sdk /opt/xcelsior /opt/xcelsior
ENV PYTHONPATH=/opt/xcelsior
ENV XCELSIOR_HTTP_PORT=8080
```

---

## Repository Layout

- [`Dockerfile`](./Dockerfile) — image definition for `ghcr.io/xcelsior-gpu/serverless-base:py3.12`
- [`serverless/worker_sdk/`](./serverless/worker_sdk) — Python Worker SDK source
- [`examples/echo/`](./examples/echo) — minimal worker built and smoke-tested on every release
