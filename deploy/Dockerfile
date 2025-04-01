FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-dev python3-tk build-essential tzdata curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages (optimized for CUDA 11.8)
COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --no-cache-dir --progress-bar on --verbose

# Copy rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]

