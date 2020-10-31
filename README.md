# gcns-fyp

This repository contains the code for my Final Year Project (FYP) at Monash University, Department of Electrical and Computer Systems Engineering. The projects explores the use of Graph Convolutional Networks (GCNs), with a main focus on their application to predicting the properties of molecules.

## Installation/Running

This project uses Docker containers in order to create a reproducible environment. There is currently a signle Dockerfile for CPU-only environments, with the view of adding an additional environment that utilises GPUs/TPUs in the future. To build the `torch-cpu` container, run the following:

```
 docker build -f Dockerfile-CPU --label torch-cpu .

```
Once the image is built, a container can be created from the image with
```
docker run -it torch-cpu
```
