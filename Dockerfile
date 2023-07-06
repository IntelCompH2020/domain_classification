# Base image
ARG VARIANT=20.04
FROM ubuntu:${VARIANT}

# Non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update
RUN apt-get upgrade -y

# # NVIDIA 
# RUN apt-get install nvidia-container-runtime

# Python
RUN apt-get update && apt-get -y install --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-wheel \
    build-essential \
    python-is-python3
# RUN pip install --upgrade pip, wheel, setuptools
# X11
RUN apt-get update && apt-get -y install --no-install-recommends x11-apps
# Qt
RUN apt-get update && apt-get -y install --no-install-recommends qt5-default

# Build directory architecture
RUN mkdir /shared_data
RUN mkdir /shared_data/projects
RUN mkdir /shared_data/datasets
RUN mkdir /shared_data/models
WORKDIR /app

# Application
COPY src/ src/
COPY config/ config/
COPY run_dc_task.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

# Initialize
CMD ["bash"]
