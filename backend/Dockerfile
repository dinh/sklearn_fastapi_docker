# backend/Dockerfile
FROM python:3.8.13-slim-bullseye

WORKDIR /app

COPY requirements.txt /app
RUN python -m pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

COPY app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]