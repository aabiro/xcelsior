# syntax=docker/dockerfile:1
FROM xcelsior/serverless-base:cuda12.4-py3.12

WORKDIR /app
COPY handler.py .

ENV XCELSIOR_HTTP_PORT=8080
ENV XCELSIOR_SKIP_CUDA_CHECK=1

CMD ["python", "handler.py"]