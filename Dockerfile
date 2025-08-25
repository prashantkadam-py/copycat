# Use lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Run FastAPI app
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT

