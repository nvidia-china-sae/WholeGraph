#docker pull nvcr.io/nvidian/sae/ydx_whole_graph_pytorch:v0.3
#docker build -t wholegraph.pytorch -f Dockerfile.pytorch .
HERE=`pwd`
docker run --gpus all -it --rm \
           --ipc=host \
           -v $HERE:$HERE \
    	   -v /home/xiaonans/projects/gnn/wholegraph/dataset:/home/xiaonans/projects/gnn/wholegraph_git/dataset \
           -w $HERE \
	       -u $(id -u):$(id -g) \
           wholegraph.pytorch

