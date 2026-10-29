# Serverless worker quickstart

## 1. Echo handler (queue mode)

```bash
# From repo root — build base image
docker build -f serverless/images/base/Dockerfile -t xcelsior/serverless-base:cuda12.4-py3.12 .

# Build echo example
docker build -f serverless/examples/echo/Dockerfile -t my-echo-worker .
```

Create a **custom** endpoint via API with `image_ref: my-echo-worker`, then invoke:

```bash
curl -X POST "$API/v1/serverless/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "hello"}}'
```

## 2. Managed preset (vLLM)

Platform default image: `vllm/vllm-openai:v0.8.5`, the published upstream
OpenAI server. The platform passes engine arguments to its entrypoint, binds
the configured HTTP port (8080 by default), and sets the Hugging Face cache
under `/xcelsior-cache/huggingface`.

```bash
docker manifest inspect vllm/vllm-openai:v0.8.5
```

The old `xcelsior/serverless-vllm:12.4` image was not published. Existing
endpoints retaining that image need an explicit image and startup-command
update before restarting; changing the default does not rewrite them.
`serverless/images/vllm/Dockerfile` remains available for a custom build that
you publish to your own registry.

Create a **preset** endpoint with `model_name: meta-llama/Llama-3.1-8B-Instruct`.
OpenAI SDK base URL:

```
https://api.xcelsior.ca/v1/serverless/{endpoint_id}/openai/v1
```

## 3. Local handler dev (no GPU)

```python
import os
os.environ["XCELSIOR_SKIP_CUDA_CHECK"] = "1"

from serverless.worker_sdk import handler, run_worker

@handler
def process(job):
    return {"ok": True, "input": job["input"]}

# run_worker(skip_fitness=True) in tests; production runs full checks
```

## Billing note

Jobs are metered by **GPU-seconds** from `started_at` → `finished_at` on the
control plane. Worker uptime (scale-to-zero hold) is billed separately via the
background `serverless_workers` tick. See `INFERENCE_DESIGN.md` §8.
