from wholegraph.common.basics import WholeGraphBasics as _WholeGraphBasics
from wholegraph.common.basics import OGBConvertedHomoGraphConfig, DistributedHomoGraph, get_ext_suffix
import ctypes

import horovod.torch as hvd
import os

import torch

cwd = os.getcwd()
so_path = os.path.join(cwd, 'wholegraph/torch/__init__.py')
so_name = 'wholegraph_pytorch_wrapper'
_basics = _WholeGraphBasics(so_path, so_name)
# _basics = _WholeGraphBasics(__file__, 'wholegraph_pytorch_wrapper')


def init():
    hvd.init()
    if not hvd.mpi_enabled():
        raise ModuleNotFoundError("MPI not enabled for Horovod")
    _basics.init()
    dir_path = os.path.dirname(so_path)
    full_path = os.path.join(dir_path, so_name + get_ext_suffix())
    torch.ops.load_library(full_path)


shutdown = _basics.shutdown
is_initialized = _basics.is_initialized
create_homograph_from_ogb_graph = _basics.create_homograph_from_ogb_graph
destroy_homograph = _basics.destroy_homograph


def load_ogb_homograph(converted_graph_dir, directed):
    graph_config = OGBConvertedHomoGraphConfig()
    graph_config.converted_dir = ctypes.cast(
        ctypes.create_string_buffer(converted_graph_dir.encode('utf-8')), ctypes.c_char_p)
    graph_config.directed = directed

    dist_graph = _basics.create_homograph_from_ogb_graph(graph_config)
    return dist_graph



# test functions
generate_test_homograph = _basics.generate_test_homograph
run_homograph_test = _basics.run_homograph_test
