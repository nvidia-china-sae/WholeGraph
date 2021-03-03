#
# The MIT License (MIT)
# 
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 PYTHONPATH=.. horovodrun --disable-cache -np 16 -H localhost:16 --mpi-args="--oversubscribe" \
	    python3 ../examples/torch/ogbn_labelreuse.py \
        -g ../dataset/ogbn_papers100M/converted  -u -b 512 -w 0 -e 200 --uselabel -r 1 --weightdecay 1e-5 \
        -t 100 -l 3 -p 10 -n '16,16,16' -s '16,16,16' --optimizer 'adam' --lr 0.001 --neighbordropout 0.3 \
        -d 0.3 --hiddensize 256 --embeddingdropout 0.1 --heads 8 -o 'paper' \
        --normadj

