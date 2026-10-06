FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY project/requirements.txt /app/project/requirements.txt
RUN pip install --no-cache-dir -r /app/project/requirements.txt

COPY project/ /app/project/

EXPOSE 8002

CMD ["uvicorn", "project.server:app", "--host", "0.0.0.0", "--port", "8002"]
