FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# no interactive. kimdohyun
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN apt-get install unzip
#RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
# kimdohyun
RUN pip3 install numpy==1.20.3
RUN pip3 install scipy matplotlib
# kimdohyun
RUN pip3 install pyyaml==5.4.1
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN pip3 install jsonpickle==0.9.6
RUN pip3 install setuptools
# kimdohyun
RUN pip3 install sacred==0.7.5
# kimdohyun
RUN pip3 install pymongo tensorboardX
# install Sacred (from OxWhirl fork)
# RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

## -- SMAC
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /workspace/3rdparty/StarCraftII

WORKDIR /workspace

RUN pip3 install opencv-python gymnasium
RUN apt-get install -y libgl1-mesa-glx
