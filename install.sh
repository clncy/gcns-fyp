export TORCH_VERSION=1.8.0
export CUDA_VERSION=cu111

# Update apt
sudo apt update
sudo apt -y upgrade

# Create conda environment
conda update -y conda
conda create -y --name autograph python=3.8
conda activate autograph

# Install pytorch
python -m pip install torch==${TORCH_VERSION}+${CUDA_VERSION} torchvision==0.9.0+${CUDA_VERSION} torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch geometric
python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-geometric

# Install Rdkit for MoleculeNet datasets
conda install -y -c conda-forge rdkit