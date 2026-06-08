# syntax=docker/dockerfile:1
# xcelsior/serverless-vllm:12.4 — managed preset (vLLM OpenAI server)
FROM vllm/vllm-openai:latest

ENV HF_HOME=/xcelsior-cache/huggingface
ENV TRANSFORMERS_CACHE=/xcelsior-cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/xcelsior-cache/huggingface

# Platform injects startup_command, e.g.:
#   --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 \
#   --chat-template-content-format openai

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -sf http://127.0.0.1:8080/health || exit 1