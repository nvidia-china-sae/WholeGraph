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

## Our results:
total params: 713,754


train_loss	valid_acc 	test_acc 

0.8121

0.7112	0.6696

0.8099	0.7114	0.6713

0.8066	0.7113	0.6684

0.8088	0.7108	0.6703

0.8068	0.7113	0.6694

0.8218	0.7110	0.6693

0.8157	0.7111	0.6689

0.8068	0.7111	0.6697

0.8113	0.7111	0.6686

0.8170	0.7107	
0.6676

mean	0.8117	0.7111	0.6693
std	0.0051	0.0002	0.0010

