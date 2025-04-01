FROM ubuntu:22.04

# Install core system packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget curl gnupg lsb-release software-properties-common \
    python3 python3-pip python3-dev python3-tk build-essential python3-numpy \
    tzdata && rm -rf /var/lib/apt/lists/*

# Set up CUDA 11.8 APT repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --no-cache-dir --progress-bar on --verbose

# Copy app source code
COPY . .

# Set environment variables
ENV FLASK_APP=app.py

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
