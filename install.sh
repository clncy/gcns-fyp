TORCH=1.8.0
CUDA=cu111

# Install python
sudo apt install python3.8

# Install pytorch
pip install torch==${TORCH}+${CUDA} torchvision==0.9.0+${CUDA} torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch geometric
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-geometric


