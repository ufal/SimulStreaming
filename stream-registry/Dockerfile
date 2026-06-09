FROM python:3.12-slim

WORKDIR /app

COPY stream-registry/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY stream-registry /app/stream-registry

ENV PYTHONUNBUFFERED=1
ENV PORT=8003

EXPOSE 8003

CMD ["sh", "-lc", "uvicorn stream-registry.main:app --host 0.0.0.0 --port ${PORT}"]
