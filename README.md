## Prerequisites

- **Hardware:** [DGX Station™ A100](https://www.nvidia.com/en-us/data-center/dgx-station-a100/)
- **Docker:** Docker CE v19.03+ and [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
- **NVIDIA Drivers:** 450.80+

## Environment Setup

- Download this source codes:
```
git clone https://gitlab-master.nvidia.com/xiaonans/wholegraph_github.git
```

- Build docker from Dockerfile.pytorch:
```
cd wholegraph_github
sh docker.sh
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
