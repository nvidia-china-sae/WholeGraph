## Prerequisites

- **Hardware:** [DGX Station™ A100](https://www.nvidia.com/en-us/data-center/dgx-station-a100/)
- **Docker:** Docker CE v19.03+ and [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
- **NVIDIA Drivers:** 450.80+

## Environment Setup

- Download this source codes:
```
git clone https://gitlab-master.nvidia.com/xiaonans/wholegraph_github.git
```

- Build docker from Dockerfile.pytorch and run the container:
```
cd wholegraph_github
sh docker.sh
```

- Download the ogbn-papers100M dataset and convert it to binary format:
```
mkdir dataset
cd build
python3 ../examples/tools/ogb_data_convert.py -d ogbn-papers100M -r ../dataset
```

## Run
Make sure you are under the _build_ dir.
```
sh run.sh
```
