## Prerequisites

- **Hardware:** [NVIDIA DGX A100 320GB](https://www.nvidia.com/en-us/data-center/dgx-a100/) or [NVIDIA DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2/)
- **Docker:** Docker CE v19.03+ and [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
- **NVIDIA Drivers:** 450.80+

## Environment Setup

- Download the source codes:
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

For the score of "gnn_benchmark", run the following command.
```
sh run_benchmark.sh
```

For the score of "gnn_addfeature", run the following command.
```
sh run_addfeature.sh
```

Running summaries can be got at result/papers100m.

## Our results:
Method | Test Accuracy | Valid Accuracy | Parameters | Hardware
| ------ | ------ | ------ | ------ | ------ |
| gnn_benchmark | 0.6693 ± 0.0010 | 0.7111 ± 0.0002 | 713,754 | 7*A100(40GB) |
| gnn_addfeature | 0.6736 ± 0.0010 | 0.7172 ± 0.0005 | 883,378 | 16*V100(32GB) |
 


