# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модель
COPY . .
COPY ../reports/model/ ./reports/model/

# Запуск
CMD ["python", "app.py"]