docker build -t wholegraph.pytorch -f Dockerfile.pytorch .
HERE=`pwd`
docker run --gpus all -it --rm --privileged=true \
           -v $HERE:$HERE \
           -w $HERE \
           wholegraph.pytorch

