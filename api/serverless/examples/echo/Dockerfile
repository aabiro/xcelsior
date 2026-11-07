# syntax=docker/dockerfile:1
FROM ghcr.io/xcelsior-gpu/serverless-base:py3.12

WORKDIR /app
COPY handler.py .

ENV XCELSIOR_HTTP_PORT=8080
ENV XCELSIOR_SKIP_CUDA_CHECK=1

CMD ["python", "handler.py"]
