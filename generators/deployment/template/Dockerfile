FROM nvidia/cuda:10.0-base-ubuntu16.04 as builder

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app

RUN curl -so ~/Miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/Miniconda.sh -b -p /opt/conda && \
    rm ~/Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
    torchvision=0.2.1 \
 && conda clean -ya

COPY deployment/template/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
ADD https://storage.googleapis.com/questo2.appspot.com/template/resources.zip /app
RUN python -m spacy download en_core_web_lg

COPY ./ /app
WORKDIR /app
RUN unzip resources.zip
ENTRYPOINT ["python3", "TemplatedQuestions.py"]

