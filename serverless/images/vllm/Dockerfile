# syntax=docker/dockerfile:1
# xcelsior/serverless-vllm:12.4 — managed preset (vLLM OpenAI server)
#
# Pin vLLM (was :latest) to a reproducible recent build. Must be >= 0.6 so
# embedding (--task embed → /v1/embeddings) and reranker (--task score →
# /v1/rerank, /v1/score) presets work, and new models (Llama 3.3, Qwen2.5-Coder,
# DeepSeek-R1 distills) are supported. Bump VLLM_VERSION to roll forward.
ARG VLLM_VERSION=v0.8.5
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV HF_HOME=/xcelsior-cache/huggingface
ENV TRANSFORMERS_CACHE=/xcelsior-cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/xcelsior-cache/huggingface

# Platform injects startup_command, e.g.:
#   --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096 \
#   --chat-template-content-format openai

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -sf http://127.0.0.1:8080/health || exit 1