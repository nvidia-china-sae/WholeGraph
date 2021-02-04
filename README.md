## Dependencies
Make sure you have installed the[ NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and Docker engine for your Linux distribution Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed.
For instructions on getting started with the NVIDIA Container Toolkit, refer to [the installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Environment Setup

If using TensorFlow version, build docker from Dockerfile.tensorflow, or use [this docker image](nvcr.io/nvidian/sae/ydx_whole_graph:v0.3).

If using PyTorch version, build docker from Dockerfile.pytorch, or use [this docker image](nvcr.io/nvidian/sae/ydx_whole_graph_pytorch:v0.3).

## Build

```
mkdir build
cd build
cmake ../
make -j
```

## Run

To run the sample without install, from the build directory:

First convert OGB data to binary format.
```
python3 ../examples/tools/ogb_data_convert.py -d [DATASET] -r [ROOTDIR]
```

TensorFlow version:
```
PYTHONPATH=.. horovodrun --disable-cache  -np 4 -H localhost:4 python3 ../examples/tensorflow/simple_test.py -g [CONVERTED_DIR]
```
PyTorch version:
```
PYTHONPATH=.. horovodrun --disable-cache  -np 4 -H localhost:4 python3 ../examples/torch/simple_test.py -g [CONVERTED_DIR]
```
