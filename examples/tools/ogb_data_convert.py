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

from optparse import OptionParser
import os
import numpy as np
import yaml
import pickle


_supported_datasets = {
    'node': ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv', 'ogbn-papers100M', 'ogbn-mag'],
    'edge': ['ogbl-ppa', 'ogbl-collab', 'ogbl-ddi', 'ogbl-citation', 'ogbl-wikikg', 'ogbl-biokg'],
    'graph': ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code'],
}


def check_is_supported_dataset(task_type, dataset_name):
    if task_type not in _supported_datasets.keys():
        raise ValueError('Unsupported task_type %s' % task_type)
    if dataset_name not in _supported_datasets[task_type]:
        raise ValueError('Unsupported dataset %s for task_type %s' % (dataset_name, task_type))
    print('Processing dataset %s of task type %s' % (dataset_name, task_type))


def dataset_name_to_dir_name(dataset_name):
    return '_'.join(dataset_name.split('-'))


def convert_node_prop_dataset(d_name, root_dir):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name=d_name, root=root_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    dir_name = dataset_name_to_dir_name(d_name)
    for name in ['num_nodes', 'edge_index', 'node_feat', 'edge_feat']:
        if name not in graph.keys():
            raise ValueError('graph has no key %s, graph.keys()= %s' % (name, graph.keys()))
    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index']
    node_feat = graph['node_feat']
    edge_feat = graph['edge_feat']
    if isinstance(num_nodes, np.int64) or isinstance(num_nodes, np.int32):
        num_nodes = num_nodes.item()
    if not isinstance(edge_index, np.ndarray) or len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
        raise TypeError('edge_index is not numpy.ndarray of shape (2, x)')
    num_edges = edge_index.shape[1]
    node_feat_dim = 0
    if node_feat is not None:
        if not isinstance(node_feat, np.ndarray) or len(node_feat.shape) != 2 or node_feat.shape[0] != num_nodes:
            raise ValueError('node_feat is not numpy.ndarray of shape (num_nodes, x)')
        node_feat_dim = node_feat.shape[1]
    edge_feat_dim = 0
    if edge_feat is not None:
        if not isinstance(edge_feat, np.ndarray) or len(edge_feat.shape) != 2 or edge_feat.shape[0] != num_edges:
            raise ValueError('edge_feat is not numpy.ndarray of shape (num_edges, x)')
        edge_feat_dim = edge_feat.shape[1]

    yaml_dict = {'num_nodes': num_nodes,
                 'num_edges': num_edges,
                 'node_feat_dim': node_feat_dim,
                 'edge_feat_dim': edge_feat_dim}
    output_dir = os.path.join(root_dir, dir_name, 'converted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'meta.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f)
    with open(os.path.join(output_dir, 'edge_index.bin'), "wb") as f:
        edge_index.tofile(f)
    if node_feat is not None:
        with open(os.path.join(output_dir, 'node_feat.bin'), "wb") as f:
            node_feat.tofile(f)
    if edge_feat is not None:
        with open(os.path.join(output_dir, 'edge_feat.bin'), "wb") as f:
            edge_feat.tofile(f)
    train_label = label[train_idx]
    valid_label = label[valid_idx]
    test_label = label[test_idx]
    data_and_label = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx,
                      'train_label': train_label, 'valid_label': valid_label, 'test_label': test_label}
    with open(os.path.join(output_dir, 'data_and_label.pkl'), "wb") as f:
        pickle.dump(data_and_label, f)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-t', '--type', dest='task_type', default='node', metavar="TYPE",
                      help="dataset task type, valid values are node, edge or graph, now only Node supported.")
    parser.add_option('-d', '--dataset', dest='dataset_name', default='ogbn-arxiv', metavar="DATASET",
                      help="dataset name")
    parser.add_option('-r', '--root', dest='root_dir', default='dataset', metavar='ROOTDIR',
                      help='root directory for dataset')
    (options, args) = parser.parse_args()
    check_is_supported_dataset(options.task_type, options.dataset_name)
    if options.task_type == 'node':
        convert_node_prop_dataset(options.dataset_name, options.root_dir)
    else:
        raise NotImplementedError('task type edge and graph are not implemented now.')
