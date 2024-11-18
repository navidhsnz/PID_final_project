#FROM pytorch/pytorch
FROM nvidia/opengl:1.2-glvnd-devel

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update -y && apt-get install -y  \
    freeglut3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    wget curl vim git \
    && \
    rm -rf /var/lib/apt/lists/*

#Installing ocnda 
RUN arch=$(uname -m) && \
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

WORKDIR /gym-duckietown

COPY . .

RUN conda env create -f environment.yaml
RUN conda init
RUN . /root/.bashrc
RUN conda activate gym-duckietown

RUN apt update

RUN pip install -e .
RUN pip install numpy==1.22
RUN pip install pyglet==1.5.15
RUN pip install Pillow
RUN apt install xvfb
RUN conda install -c conda-forge libstdcxx-ng 