import atexit
import ctypes
import os
import sysconfig


# When adding data in this class, DON'T forget to do the same thing in distributed_graph.h
class OGBConvertedHomoGraphConfig(ctypes.Structure):
    """Wrapper class for Graph data converted from OGB"""
    def __init__(self, **kwargs):
        """
        Ctypes.Structure with integrated default values.

        :param kwargs: values different to defaults
        :type kwargs: dict
        """
        values = type(self)._defaults_.copy()
        for (key, val) in kwargs.items():
            values[key] = val
        super(OGBConvertedHomoGraphConfig, self).__init__(**values)

    _fields_ = [("converted_dir", ctypes.c_char_p),
                ("directed", ctypes.c_bool),
                ("need_in_degree", ctypes.c_bool),
                ("need_out_degree", ctypes.c_bool)]
    _defaults_ = {"directed": True, "need_in_degree": True, "need_out_degree": True}


class DistributedHomoGraph(object):
    """Wrapper class for Homo Graph object"""
    def __init__(self):
        self.node_edge_offset = None
        self.edge_dst_node_ngid = None
        self.node_feat = None
        self.edge_egid = None
        self.edge_feat = None
        self.num_nodes = 0
        self.num_edges = 0
        self.node_feat_dim = 0
        self.edge_feat_dim = 0
        self._c_graph = None
        self.LIB_CTYPES = None

    def __str__(self):
        return 'DistributedHomoGraph with num_node=%d, num_edges=%d, node_feat_dim=%d, edge_feat_dim=%d' % (
        self.num_nodes, self.num_edges, self.node_feat_dim, self.edge_feat_dim)

    @property
    def c_graph(self):
        return self._c_graph

    @c_graph.setter
    def c_graph(self, c_graph):
        if isinstance(c_graph, ctypes.c_void_p):
            self._c_graph = c_graph
        else:
            raise ValueError("c_graph is Illegal.")

    @c_graph.deleter
    def c_graph(self):
        del self._c_graph

    def get_node_value_handle(self, name):
        name_charp = ctypes.cast(ctypes.create_string_buffer(name.encode('utf-8')), ctypes.c_char_p)
        return ctypes.cast(self.LIB_CTYPES.wholegraph_get_homograph_node_value_handle(self.c_graph, name_charp), ctypes.c_void_p)

    def add_int_node_value(self, name, raw_id_np, value_np, default_value, bit_width=32):
        c_default_value = ctypes.c_int64(default_value)
        assert raw_id_np.size == value_np.size
        name_charp = ctypes.cast(ctypes.create_string_buffer(name.encode('utf-8')), ctypes.c_char_p)
        self.LIB_CTYPES.whograph_add_homograph_int_node_value(self.c_graph, name_charp, raw_id_np.ctypes.data_as(ctypes.c_void_p), value_np.ctypes.data_as(ctypes.c_void_p), raw_id_np.size, c_default_value, bit_width)

    def add_float_node_value(self, name, raw_id_np, value_np, default_value, bit_width=32):
        c_default_value = ctypes.c_double(default_value)
        assert raw_id_np.size == value_np.size
        name_charp = ctypes.cast(ctypes.create_string_buffer(name.encode('utf-8')), ctypes.c_char_p)
        self.LIB_CTYPES.whograph_add_homograph_float_node_value(self.c_graph, name_charp, raw_id_np.ctypes.data_as(ctypes.c_void_p), value_np.ctypes.data_as(ctypes.c_void_p), raw_id_np.size, c_default_value, bit_width)


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


class WholeGraphBasics(object):
    """Wrapper class for the basic WholeGraph API."""
    def __init__(self, pkg_path, *args):
        dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
        full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
        self.LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
        self.LIB_CTYPES.wholegraph_create_homograph_from_ogb_graph.restype = ctypes.c_void_p
        self.LIB_CTYPES.wholegraph_get_homograph_node_value_handle.restype = ctypes.c_void_p
        #self.LIB_CTYPES.wholegraph_gather_node_feature.restype = ctypes.POINTER(ctypes.c_float)

    def init(self):
        """A function that initializes WholeGraph."""
        #atexit.register(self.shutdown)
        self.LIB_CTYPES.wholegraph_init()

    def shutdown(self):
        """A function that shuts WholeGraph down."""
        self.LIB_CTYPES.wholegraph_shutdown()

    def is_initialized(self):
        """Returns True if WholeGraph is initialized"""
        return self.LIB_CTYPES.wholegraph_is_initialized()

    def create_homograph_from_ogb_graph(self, graph_cfg):
        if not isinstance(graph_cfg, OGBConvertedHomoGraphConfig):
            raise ValueError("graph_cfg is not OGBConvertedHomoGraphConfig.")
        c_graph = ctypes.c_void_p(self.LIB_CTYPES.wholegraph_create_homograph_from_ogb_graph(ctypes.byref(graph_cfg)))
        if c_graph:
            peer_mem_frag_array = (ctypes.c_void_p * 5)()
            node_edge_info = (ctypes.c_int64 * 4)()
            self.LIB_CTYPES.wholegraph_get_homograph_info(peer_mem_frag_array, node_edge_info, c_graph)
            dist_graph = DistributedHomoGraph()
            dist_graph.c_graph = c_graph

            dist_graph.node_edge_offset = ctypes.cast(peer_mem_frag_array[0], ctypes.c_void_p)
            dist_graph.edge_dst_node_ngid = ctypes.cast(peer_mem_frag_array[1], ctypes.c_void_p)
            dist_graph.node_feat = ctypes.cast(peer_mem_frag_array[2], ctypes.c_void_p)
            dist_graph.edge_egid = ctypes.cast(peer_mem_frag_array[3], ctypes.c_void_p)
            dist_graph.edge_feat = ctypes.cast(peer_mem_frag_array[4], ctypes.c_void_p)

            dist_graph.num_nodes = node_edge_info[0]
            dist_graph.num_edges = node_edge_info[1]
            dist_graph.node_feat_dim = node_edge_info[2]
            dist_graph.edge_feat_dim = node_edge_info[3]

            dist_graph.LIB_CTYPES = self.LIB_CTYPES

            return dist_graph
        else:
            raise RuntimeError("Create Graph failed.")

    def destroy_homograph(self, dist_graph):
        if isinstance(dist_graph, DistributedHomoGraph):
            if isinstance(dist_graph.c_graph, ctypes.c_void_p):
                success = self.LIB_CTYPES.wholegraph_destroy_homograph(dist_graph.c_graph)
                if not success:
                    raise RuntimeError("Destroy graph failed.")
                del dist_graph.c_graph
            else:
                raise ValueError("dist_graph.c_graph not ctypes.c_void_p")
        else:
            raise ValueError("dist_graph not DistributedHomoGraph")


    def generate_test_homograph(self, node_count, min_degree, max_degree, node_feat_dim, edge_feat_dim, output_path='dataset/generated/'):
        # wg.generate_test_homograph(5000, 5, 30, 128, 128, '/home/dongxuy/dataset/gnn/dataset/generated/converted')
        # wg.generate_test_homograph(4, 1, 2, 3, 3, '/home/dongxuy/dataset/gnn/dataset/generated/converted')
        param_node_count = ctypes.c_int64(node_count)
        param_min_degree = ctypes.c_int64(min_degree)
        param_max_degree = ctypes.c_int64(max_degree)
        param_node_feat_dim = ctypes.c_int64(node_feat_dim)
        param_edge_feat_dim = ctypes.c_int64(edge_feat_dim)
        param_output_path = ctypes.cast(ctypes.create_string_buffer(output_path.encode('utf-8')), ctypes.c_char_p)
        self.LIB_CTYPES.wholegraph_generate_test_homograph(param_node_count, param_min_degree, param_max_degree, param_node_feat_dim, param_edge_feat_dim, param_output_path)

    def run_homograph_test(self, dist_graph, min_degree, max_degree):
        param_min_degree = ctypes.c_int64(min_degree)
        param_max_degree = ctypes.c_int64(max_degree)
        if isinstance(dist_graph, DistributedHomoGraph):
            if isinstance(dist_graph.c_graph, ctypes.c_void_p):
                self.LIB_CTYPES.wholegraph_run_homograph_test(dist_graph.c_graph, param_min_degree, param_max_degree)
            else:
                raise ValueError("dist_graph.c_graph not ctypes.c_void_p")
        else:
            raise ValueError("dist_graph not DistributedHomoGraph")
