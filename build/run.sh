CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 PYTHONPATH=.. horovodrun --disable-cache -np 7 -H localhost:7 \
	python3 ../examples/torch/papers100m_benchmark.py \
	-g '../dataset/ogbn_papers100M/converted' -b 512 -e 200 -w 0  -s 16 -n 16 -u -o 'papers100m' \
	--uselabel -r 10 --hiddensize 256 -t 100 --lr 0.01
