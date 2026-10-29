# Serverless worker handler contract

Queue-mode custom endpoints run a Python handler inside `xcelsior/serverless-base`.
The worker SDK polls the control plane, executes jobs, and streams progress events.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XCELSIOR_API_URL` | yes | Control plane base URL |
| `XCELSIOR_WORKER_ID` | yes | `swkr-…` row id |
| `XCELSIOR_API_TOKEN` | prod | Agent/worker auth token |
| `XCELSIOR_HTTP_PORT` | no | Health server port (default `8080`) |
| `XCELSIOR_HOST_ID` | no | Host id reported on ready |
| `HF_HOME` | preset | NFS cache mount for model weights |

## Handler signature

```python
from serverless.worker_sdk import handler, run_worker

@handler
def process(job: dict) -> dict:
    prompt = job["input"].get("prompt", "")
    return {"text": prompt.upper()}

if __name__ == "__main__":
    run_worker()
```

Supported return types:

- `dict` — job output (stored on `serverless_jobs.output`)
- `str` / `int` / `float` — wrapped as `{"result": value}`
- `generator` / `async generator` — yields progress/log/output events, final value is output

## Progress events

```python
from serverless.worker_sdk import progress_event, log_event

yield progress_event("Loading model", pct=25)
yield log_event("Tokenizing input")
```

Events are persisted to `serverless_job_stream_events` and replayed over SSE.

## Error envelope

Raise `WorkerError` for structured failures:

```python
from serverless.worker_sdk import WorkerError

raise WorkerError(422, "invalid_input", "prompt is required", retryable=False)
```

## ASGI mode (custom OpenAI-native images)

For containers that already expose an HTTP server (e.g. `vllm/vllm-openai`):

```python
from fastapi import FastAPI
from serverless.worker_sdk import serve_asgi

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

serve_asgi(app)
```

Configure the endpoint `http_port` and `health_check_path` to match the container.

## Control-plane callbacks

| Route | Purpose |
|-------|---------|
| `POST /api/v2/serverless/workers/{id}/ready` | Worker passed fitness + health |
| `POST …/heartbeat` | Liveness + idle/ready state |
| `POST …/jobs/claim` | Claim next queued job (`SKIP LOCKED`) |
| `POST …/jobs/{job_id}/complete` | Finish job + release concurrency |
| `POST …/jobs/{job_id}/events` | Append stream event |

## Fitness checks

On boot, `run_worker()` verifies (unless skipped):

- CUDA GPU available (`nvidia-smi` or PyTorch)
- Minimum free disk and memory
- Optional `XCELSIOR_WEIGHTS_READY_PATH` file exists

Set `XCELSIOR_SKIP_CUDA_CHECK=1` for CPU-only dev images.

## Deploy resilience

Serverless follows the same deploy patterns as interactive instances:

- **API blue-green:** nginx `proxy_next_upstream` fails over to the standby API
  colour; worker callbacks retry transient 502/503/504 errors automatically.
- **Long streams:** `/v1/serverless/{id}/stream/…` and streaming OpenAI chat use
  dedicated nginx locations with `proxy_buffering off` and 24h read timeout.
- **Worker host maintenance:** scheduler jobs use `job_type=serverless_worker` and
  block `--guard-host` until drained (same as web terminal instances).
- **State:** endpoints, workers, and queued jobs live in Postgres — survive API
  container restarts; `bg_worker` reconcile resumes autoscaler ticks.