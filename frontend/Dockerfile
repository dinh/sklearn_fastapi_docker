# frontend/Dockerfile
FROM python:3.13.0a4-slim

WORKDIR /app
ENV FLASK_APP=main.py
COPY requirements.txt /app
RUN python -m pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt
COPY app /app
# CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]
