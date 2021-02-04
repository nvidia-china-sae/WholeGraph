# PREREQUISITES
## Hardware: [DGX Station™ A100](https://www.nvidia.com/en-us/data-center/dgx-station-a100/)
## Docker: Docker CE v19.03+ and [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
## NVIDIA Drivers: 450.80+

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
