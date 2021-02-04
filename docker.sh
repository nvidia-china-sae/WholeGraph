docker build -t wholegraph.pytorch -f Dockerfile.pytorch .
HERE=`pwd`
docker run --gpus all -it --rm --privileged=true \
           --ipc=host \
           -v $HERE:$HERE \
    	   -v /raid/privatedata/pursuit/dongxuy/dataset/gnn/dataset:$HERE/dataset \
           -w $HERE \
           wholegraph.pytorch

