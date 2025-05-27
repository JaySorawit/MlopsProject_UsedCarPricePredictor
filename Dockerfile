# เลือก base image ที่ใช้ Python
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port 5000 for Flask or FastAPI
EXPOSE 5000

# start API server
CMD ["python", "app/serve_model.py"]
