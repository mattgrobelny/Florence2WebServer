# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Update package lists and install dependencies for Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y python3-pip

# Symlink 'python3' to 'python3.10'
# RUN ln -s /usr/bin/python3.10 /usr/bin/python3

# Install pip for Python 3.10
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Verify installations
# RUN python3 --version && pip3 --version

# Copy requirements.txt to /app
COPY /requirements.txt .
# Install dependencies
RUN pip3 install -r requirements.txt

RUN pip3 install uvicorn opencv-python
RUN pip3 install timm einops
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install python-multipart

# Download model during the build process (using python script to ensure it's cached)
RUN python3 -c "\
import torch; \
from transformers import AutoProcessor, AutoModelForCausalLM; \
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; \
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device); \
print('Model and processor downloaded and cached.')"

# Expose the port for the FastAPI server
EXPOSE 8000
# ARG PORT=${PORT}
# EXPOSE ${PORT}

COPY /main.py .
COPY /FlorenceModel.py .

# Set the environment variable for CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Pass down API key from environment variable
ENV API_KEY=${API_KEY}

# Run the command to start the FastAPI server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]

