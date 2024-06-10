# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions

FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies
RUN python3 -m pip install autopep8
RUN python3 -m pip install attrdict
RUN python3 -m pip install h5py
RUN python3 -m pip install jsonlines
RUN python3 -m pip install rich
RUN python3 -m pip install wandb
RUN python3 -m pip install plotly
RUN python3 -m pip install pytablewriter

# Install additional dependencies
RUN python3 -m pip install transformers>=4.26.0
RUN python3 -m pip install datasets>=1.8.0
RUN python3 -m pip install adapter==0.1.0
RUN python3 -m pip install evaluate
RUN python3 -m pip install accelerate
RUN python3 -m pip install protobuf
RUN python3 -m pip install fasttext
RUN python3 -m pip install nltk
RUN python3 -m pip install scikit-learn


# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

RUN mkdir -p /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
RUN mkdir /home/$USER_NAME/.local

RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]

