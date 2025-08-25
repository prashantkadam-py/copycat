# Use lightweight Python image
FROM python:3.11-slim


#install system dependencies (git, curl, etc.)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Run FastAPI app
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT

