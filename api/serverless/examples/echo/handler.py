"""Minimal echo worker — returns the job input payload."""

from serverless.worker_sdk import handler, run_worker


@handler
def process(job: dict) -> dict:
    prompt = (job.get("input") or {}).get("prompt", "")
    return {"echo": prompt or job.get("input")}


if __name__ == "__main__":
    run_worker()