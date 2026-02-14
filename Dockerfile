FROM python:3.12-slim

WORKDIR /app

# Install system deps for psycopg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure templates and migrations are present
RUN test -d templates && test -d migrations

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
