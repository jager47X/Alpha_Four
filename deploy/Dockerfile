FROM ubuntu:22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies & Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common curl wget gnupg lsb-release build-essential tzdata \
    python3.10 python3.10-dev python3.10-distutils python3-pip python3-tk python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip (ensure Python 3.10 uses it)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install NVIDIA CUDA repo and toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8

# Set CUDA paths
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .

# Install correct pip packages (with numpy < 2.0 to prevent crash)
RUN pip3 install --no-cache-dir --progress-bar on --verbose --upgrade pip && \
    pip3 install --no-cache-dir --progress-bar on --verbose -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=app.py

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
