# Adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/docker/Dockerfile

FROM nvidia/cuda:11.2.2-base-ubuntu20.04

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# PyTorch (Geometric) installation
RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda.
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python environment.
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py38 python=3.8 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install pytorch 1.8.0 with CUDA 11 support
ENV TORCH 1.8.0
ENV CUDA cu111
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric.
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-geometric

# Install rdkit for usage by MoleculeNet
#RUN conda install -y -c conda-forge rdkit
#RUN sudo apt-get update
#RUN DEBIAN_FRONTEND=noninteractive sudo apt install -y --no-install-recommends  python3-rdkit
#RUN conda create -c rdkit -n my-rdkit-env rdkit
#RUN conda activate my-rdkit-env

COPY gcns-fyp /app/gcns-fyp
WORKDIR /app/gcns-fyp

#RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
