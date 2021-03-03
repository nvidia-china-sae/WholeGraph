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

import wholegraph.torch as wg
import ctypes
import os
from optparse import OptionParser
import torch
import horovod.torch as hvd
import pickle
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import SAGEConv, TransformerConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
import sys
import datetime
import time
from horovod.torch.mpi_ops import Sum
import math
from torch import nn
import pdb
from tqdm import tqdm

graphconv_layer_dict = {'sage': (SAGEConv, True), 'transformer':(TransformerConv, 1)}

def load_data(options):
    with open(os.path.join(options.converted_graph_dir, 'data_and_label.pkl'), "rb") as f:
        data_and_label = pickle.load(f)
    train_data = {'idx': data_and_label['train_idx'], 'label': data_and_label['train_label']}
    valid_data = {'idx': data_and_label['valid_idx'], 'label': data_and_label['valid_label']}
    test_data  = {'idx': data_and_label['test_idx'],  'label': data_and_label['test_label']}
    return train_data, valid_data, test_data

def load_graph(options):
    graph_config = wg.OGBConvertedHomoGraphConfig()
    graph_config.converted_dir = ctypes.cast(ctypes.create_string_buffer(options.converted_graph_dir.encode('utf-8')), ctypes.c_char_p)
    graph_config.directed = not options.undirected
    dist_graph = wg.create_homograph_from_ogb_graph(graph_config)
    return dist_graph

def get_train_step(sample_count, epochs, batch_size, global_size):
    return sample_count * epochs // (batch_size * global_size)

def create_train_dataset(options, data_tensor_dict):
    data_loader = DataLoader(dataset=list(range(len(data_tensor_dict['idx']))),  
               batch_size=options.batchsize, shuffle=True, num_workers=options.dataloaderworkers, pin_memory=False)
    idx = torch.tensor(data_tensor_dict['idx'], dtype=torch.int64)
    return data_loader, idx

def create_valid_test_dataset(options, data_tensor_dict):
    total = len(data_tensor_dict['idx'])
    stride = int((total + hvd.size() - 1)/ hvd.size())
    start = int(hvd.rank() * stride)
    end = min((hvd.rank()+1) * stride, total)
    data_loader = DataLoader(dataset=range(start, end),  
            batch_size=options.batchsize, shuffle=False, num_workers=options.dataloaderworkers, pin_memory=False)
    idx = torch.tensor(data_tensor_dict['idx'], dtype=torch.int64)
    return data_loader, idx

@torch.no_grad()
def valid_test(dataloader, model, data_size, total_idx, handles, is_test):
    model.eval()
    total_correct = 0
    total_sample = 0
    for i, batch in enumerate(dataloader):
        idx = total_idx[batch]
        if is_test:
            logits, label = model.test(idx, handles)
        else:
            logits, label = model.validate(idx, handles)
        logp = F.log_softmax(logits, 1)
        pred = torch.argmax(logp, 1)
        correct = (pred == label).sum()
        total_correct += correct.cpu()
        total_sample += pred.shape[0]
    total_sample = hvd.allreduce(torch.tensor(total_sample), op=Sum)
    assert total_sample == data_size
    total_correct = hvd.allreduce(total_correct, op=Sum)
    accuracy = 100.0*total_correct/total_sample
    return accuracy

def valid(dataloader, model, data_size, total_idx, handles):
    return valid_test(dataloader, model, data_size, total_idx, handles, False)

def test(dataloader, model, data_size, total_idx, handles):
    return valid_test(dataloader, model, data_size, total_idx, handles, True)

def train(train_dataloader, model, optimizer, total_steps, train_idx, handle):
    model.train()
    total_loss = 0
    total_sample = 0
    total_correct = 0
    for step in range(total_steps):
        optimizer.zero_grad()
        batch = next(iter(train_dataloader))
        idx = train_idx[batch]
        logits, label = model(idx, handle)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, label)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item() * batch.numel()
        total_sample = total_sample + batch.numel()
        pred = torch.argmax(logp, 1)
        correct = (pred == label).sum()
        total_correct += correct.cpu()
    total_loss = hvd.allreduce(torch.tensor(total_loss), op=Sum).item()
    total_sample = hvd.allreduce(torch.tensor(total_sample), op=Sum).item()
    total_loss = total_loss / total_sample
    total_correct = hvd.allreduce(total_correct, op=Sum)
    accuracy = 100.0*total_correct/total_sample
    return total_loss, accuracy

def run(i_run, options, train_data, valid_data, test_data, model, optimizer, handles, outfile):
    train_dataloader, train_idx = create_train_dataset(options, data_tensor_dict=train_data)
    valid_dataloader, valid_idx = create_valid_test_dataset(options, data_tensor_dict=valid_data)
    test_dataloader,  test_idx  = create_valid_test_dataset(options, data_tensor_dict=test_data)
    total_steps = get_train_step(len(train_data['idx']), 1, options.batchsize, hvd.size())
    train_step = 0
    best={'val_acc':0.0, 'epoch':0}

    for g in optimizer.param_groups:
        g['lr'] = options.lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode='max', factor=0.1, patience=options.lr_patience, threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=1e-2, eps=1e-08, verbose=True)
    
    model.reset_parameters()
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    train_epoch = 0
    for epoch in range(options.epochs):
        # train
        t0 = time.time()
        total_cla_loss, train_acc = train(train_dataloader, model, optimizer, total_steps, train_idx, handles.train_label_handle)
        t1 = time.time()        
                
        #valid
        valid_acc = valid(valid_dataloader, model, len(valid_idx), valid_idx, handles)
        t2 = time.time()
        scheduler.step(valid_acc)
        
        if valid_acc < best['val_acc']:
            if hvd.rank() == 0:
                print('run=%02d, epoch=%03d, loss=%.4f, train_acc=%.2f%%, valid_acc=%.2f%%,                  train_time=%.2fs, valid_time=%.2fs' 
                    % (i_run, epoch, total_cla_loss, train_acc, valid_acc, t1-t0, t2-t1))
            if epoch > best['epoch'] + options.stop_patience:
                break
        else:
            #test
            test_acc = test(test_dataloader, model, len(test_idx), test_idx, handles)
            t3 = time.time()
            if hvd.rank() == 0:
                print('run=%02d, epoch=%03d, loss=%.4f, train_acc=%.2f%%, valid_acc=%.2f%%, test_acc=%.2f%%, train_time=%.2fs, valid_time=%.2fs, test_time=%.2fs' 
                    % (i_run, epoch, total_cla_loss, train_acc, valid_acc, test_acc, t1-t0, t2-t1, t3-t2))
            best['val_acc'] = valid_acc
            best['loss'] = total_cla_loss
            best['test_acc'] = test_acc
            best['epoch'] = epoch
            best['train_acc'] = train_acc
        hvd.allreduce(torch.tensor(0))

    if hvd.rank() == 0:
        print('[BEST] run=%02d, epoch=%03d, loss=%.4f, train_acc=%.2f%%, valid_acc=%.2f%%, test_acc=%.2f%%' 
                    % (i_run, best['epoch'], best['loss'], best['train_acc'], best['val_acc'], best['test_acc']))
        print('[BEST] epoch=%03d, loss=%.4f, train_acc=%.2f%%, valid_acc=%.2f%%, test_acc=%.2f%%' 
            % (best['epoch'], best['loss'], best['train_acc'], best['val_acc'], best['test_acc']), file=outfile)
        outfile.flush()
    return best

class Handles(object):
    def __init__(self, train_label_handle=None, valid_label_handle=None, test_label_handle=None, 
                        out_degree_handle=None, in_degree_handle=None):
        self.train_label_handle = train_label_handle
        self.valid_label_handle = valid_label_handle
        self.test_label_handle  = test_label_handle
        self.out_degree_handle  = out_degree_handle
        self.in_degree_handle   = in_degree_handle

class HomoGNNModel(torch.nn.Module):
    def __init__(self, dist_homo_graph, num_layer, hidden_feat_dim, class_count, max_neighboor, use_label, options, handles):
        super().__init__()
        self.homo_graph = dist_homo_graph
        in_feat_dim = dist_homo_graph.node_feat_dim
        self.num_layer = num_layer
        hidden_feat_dim = hidden_feat_dim//options.heads
        self.hidden_feat_dim = hidden_feat_dim
        self.max_neighboor = max_neighboor
        self.class_count = class_count
        # submodules
        self.gnn_layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.fc = torch.nn.Linear(hidden_feat_dim*options.heads, class_count)
        if use_label:
            in_feat_dim = in_feat_dim + class_count    
        if options.use_deg:
            in_feat_dim = in_feat_dim + 3        

        for i in range(num_layer):
            if i == 0:
                self.gnn_layers.append(graphconv_layer_dict[options.gnnconv][0](in_feat_dim, 
                                        hidden_feat_dim, heads=options.heads, 
                                        dropout=options.neighbor_dropout))
            else:
                self.gnn_layers.append(graphconv_layer_dict[options.gnnconv][0](hidden_feat_dim*options.heads, 
                                        hidden_feat_dim, heads=options.heads, 
                                        dropout=options.neighbor_dropout))
            if options.regularization == 'bn':
                self.bns.append(torch.nn.BatchNorm1d(hidden_feat_dim*options.heads))
            elif options.regularization == 'ln':
                self.bns.append(torch.nn.LayerNorm(hidden_feat_dim*options.heads))
        self.dropout = torch.nn.Dropout(options.dropout)
        self.embedding_dropout = torch.nn.Dropout(options.embedding_dropout)
        self.use_label = use_label
        self.inferencesample = options.inferencesample
        self.batch_size = options.batchsize
        self.use_deg = options.use_deg
        self.handles = handles
        self.feat_bn = torch.nn.BatchNorm1d(3, affine=False)

    def reset_parameters(self):
        for conv in self.gnn_layers:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.fc.reset_parameters()

    def gnn_layer_id_gen(self, i_layer, target_gid, target_pos):
        sample_count = self.max_neighboor[i_layer] if self.training else self.inferencesample[i_layer]
        neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids = torch.ops.wholegraph.unweighted_sample(
            target_gid,
            self.homo_graph.node_edge_offset.value,
            self.homo_graph.edge_dst_node_ngid.value,
            0, sample_count, False)

        target_and_neighboor_gid = torch.cat([target_gid, neighboor_gids_vdata])
        unique_target_and_neighboor_gid, raw_to_uniqueindex = torch.unique(target_and_neighboor_gid, sorted=False, return_inverse=True, return_counts=False)
        target_lid = raw_to_uniqueindex[:target_gid.size()[0]]
        target_pos = raw_to_uniqueindex[target_pos]
        neighboor_count = neighboor_gids_vdata.size()[0]
        neighboor_dst_unique_ids = raw_to_uniqueindex[target_gid.size()[0]:]
        neighboor_src_unique_ids = neighboor_src_lids
        edge_index = torch.cat([torch.reshape(neighboor_dst_unique_ids, (1, neighboor_count)),
                                torch.reshape(neighboor_src_unique_ids, (1, neighboor_count))])
        torch.cuda.synchronize()
        return unique_target_and_neighboor_gid, edge_index, target_lid, raw_to_uniqueindex, target_pos


    def gnn_model_fn(self, gidx, target_handle, feat_handle):
        target_pos = torch.arange(gidx.numel()).cuda()
        label = torch.ops.wholegraph.gather_value_by_gid(gidx, target_handle.value)
        num_layer = self.num_layer
        target_gids = [None] * (num_layer + 1)
        target_gids[num_layer] = gidx
        edge_indice = [None] * num_layer
        target_lids = [None] * num_layer
        raw_to_uniqueindexs = [None] * num_layer

        for i in range(num_layer - 1, -1, -1):
            target_gids[i], edge_indice[i], target_lids[i], raw_to_uniqueindexs[i], target_pos \
                    = self.gnn_layer_id_gen(i, target_gids[i + 1], target_pos)
        x_feat = torch.ops.wholegraph.global_gather(target_gids[0], self.homo_graph.node_feat.value, self.homo_graph.node_feat_dim)

        #if not label_reuse and 
        if self.use_label:
            labels = torch.ops.wholegraph.gather_value_by_gid(target_gids[0], feat_handle.value)
            labels[target_pos] = self.class_count
            label_feat = torch.zeros([x_feat.size(0), self.class_count], device='cuda')
            mask = (labels != self.class_count)
            label_feat[mask, labels[mask]] = 1
            x_feat = torch.cat([x_feat, label_feat], dim=-1)
        if self.use_deg:
            out_degree = torch.ops.wholegraph.gather_value_by_gid(target_gids[0], handles.out_degree_handle.value).float().unsqueeze(1)
            in_degree = torch.ops.wholegraph.gather_value_by_gid(target_gids[0], handles.in_degree_handle.value).float().unsqueeze(1)
            degs = (out_degree + in_degree).float()
            degs = torch.cat([out_degree, in_degree, degs], dim=-1)
            deg_feat = torch.pow(degs, 0.5)/10
            x_feat = torch.cat([x_feat, deg_feat], dim=-1)
        
        x_feat = self.embedding_dropout(x_feat)
        
        torch.cuda.synchronize()
        for i in range(num_layer):
            x_target_feat = F.embedding(target_lids[i], x_feat)
            x_feat = self.gnn_layers[i]((x_feat, x_target_feat), edge_indice[i])
            x_feat = self.bns[i](x_feat)
            x_feat = F.relu(x_feat)
            x_feat = self.dropout(x_feat)
        out_feat = self.fc(x_feat)
        return out_feat, label

    def forward(self, idx, handle):
        gidx = torch.ops.wholegraph.homograph_rawid_to_globalid(idx, self.homo_graph.c_graph.value).cuda()
        return self.gnn_model_fn(gidx, handle, handle)
    
    @torch.no_grad()
    def validate(self, idx, handles):
        gidx = torch.ops.wholegraph.homograph_rawid_to_globalid(idx, self.homo_graph.c_graph.value).cuda()
        return self.gnn_model_fn(gidx, handles.valid_label_handle, handles.train_label_handle)

    @torch.no_grad()
    def test(self, idx, handles):
        gidx = torch.ops.wholegraph.homograph_rawid_to_globalid(idx, self.homo_graph.c_graph.value).cuda()
        return self.gnn_model_fn(gidx, handles.test_label_handle, handles.train_label_handle)

def create_model(dist_homo_graph, options, handles):
    return HomoGNNModel(dist_homo_graph, options.layernum, options.hiddensize, options.classnum, 
                        options.neighboor, options.uselabel, options, handles)

def main(options, dist_homo_graph, train_data, valid_data, test_data, handles):
    model = create_model(dist_homo_graph, options, handles).cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr*hvd.size(), weight_decay=options.weight_decay)
    if options.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr*hvd.size(), momentum=0.9, weight_decay=options.weight_decay)
    if options.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=options.lr*hvd.size(), momentum=0, weight_decay=options.weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    outfile = None
    if hvd.rank() == 0:
        print(options)
        if not os.path.exists(options.outdir):
            os.mkdir(options.outdir)
        outfile = open(os.path.join(options.outdir, options.outfile), 'a')
        print(options, file=outfile)
    
    best_results = []
    for i in range(options.runs):
        best = run(i, options, train_data, valid_data, test_data, model, optimizer, handles, outfile)
        best_results.append((best['loss'], best['val_acc'], best['test_acc']))

    best_result = torch.tensor(best_results)
    if hvd.rank() == 0:
        r0 = best_result[:, 0]
        r1 = best_result[:, 1]
        r2 = best_result[:, 2]
        print('[ALL RUN] train loss = %.2f ± %.2f, valid_acc = %.2f ± %.2f, test_acc=%.2f ± %.2f\n' 
            % (r0.mean(), r0.std(), r1.mean(), r1.std(), r2.mean(), r2.std()))
        print('[ALL RUN] train loss = %.2f ± %.2f, valid_acc = %.2f ± %.2f, test_acc=%.2f ± %.2f\n' 
            % (r0.mean(), r0.std(), r1.mean(), r1.std(), r2.mean(), r2.std()), file=outfile)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-g', '--converted_graph_dir', dest='converted_graph_dir', default='dataset/ogbn_arxiv/converted',
                    help="converted graph directory.")
    parser.add_option('-u', '--undirected', action="store_true", dest="undirected", default=False,
                    help='is undirected graph, default False')
    parser.add_option('-e', '--epochs', type='int', dest="epochs", default=1000,
                    help='number of epochs')
    parser.add_option('-b', '--batchsize', type='int', dest="batchsize", default=1024,
                    help='batch size')
    parser.add_option('-c', '--classnum', type='int', dest="classnum", default=40,
                    help='class number')
    parser.add_option('-n', '--neighboor', type='str',dest="neighboor", default='10,10,10',
                    help='train neighboor sample count')
    parser.add_option('-s', '--inferencesample', type='str', dest="inferencesample", default='-1,-1,-1',
                    help='inference sample count, -1 is all')
    parser.add_option('--hiddensize', type='int', dest="hiddensize", default=256,
                    help='hidden size')
    parser.add_option('-l', '--layernum', type='int', dest="layernum", default=3,
                    help='layer number')
    parser.add_option('-w', '--dataloaderworkers', type='int', dest="dataloaderworkers", default=0,
                    help='number of workers for dataloader')
    parser.add_option('-d', '--dropout', type='float', dest="dropout", default=0.3, help='dropout')
    parser.add_option('-p', '--lr_patience', type='int', dest='lr_patience', default=10, 
                    help='learning rate decay')
    parser.add_option('-t', '--stop_patience', type='int', dest='stop_patience', default=40, 
                    help='early stop')
    parser.add_option('-o', '--outfile', type='str', dest='outfile', default='arxiv', 
                    help='early stop')
    parser.add_option('--outdir', type='str', dest='outdir', default='../result', 
                    help='early stop')
    parser.add_option('--uselabel', action="store_true", dest="uselabel", default=False,
                    help='whether to add label feature')
    parser.add_option('--embeddingdropout', type='float', dest="embedding_dropout", default=0, 
                    help='embedding_dropout')
    parser.add_option('--optimizer', type='str', dest="optimizer", default='sgd', 
                    help='optimizer')
    parser.add_option('--weightdecay', type='float', dest="weight_decay", default=1e-5, 
                    help='weight decay')
    parser.add_option('--lr', type='float', dest="lr", default=0.01, 
                    help='learning rate')
    parser.add_option('-r', '--runs', type='int', dest='runs', default=10, 
                    help='number of runs')
    parser.add_option('--gnnconv', type='str', dest='gnnconv', default='transformer')
    parser.add_option('--neighbordropout', type='float', dest='neighbor_dropout', default=0)
    parser.add_option('--transformerbeta', action='store_true', dest='transformer_beta', default=False)
    parser.add_option('--regularization', type='str', dest='regularization', default='bn')
    parser.add_option('--heads', type='int', dest='heads', default=1)
    parser.add_option('--usedeg', action='store_true', dest='use_deg', default=False)


    (options, args) = parser.parse_args()

    options.neighboor = list(map(int, options.neighboor.split(',')))
    options.inferencesample = list(map(int, options.inferencesample.split(',')))


    wg.init()

    torch.cuda.set_device(hvd.local_rank())
    train_data, valid_data, test_data = load_data(options)
    num_classes = torch.Tensor(train_data['label']).unique().numel()
    options.classnum = num_classes
    dist_homo_graph = load_graph(options)
    print('Graph loaded.')

    dist_homo_graph.add_int_node_value('train_label', train_data['idx'], train_data['label'].astype(np.int64), num_classes, 64)
    dist_homo_graph.add_int_node_value('valid_label', valid_data['idx'], valid_data['label'].astype(np.int64), num_classes, 64)
    dist_homo_graph.add_int_node_value('test_label',  test_data['idx'],   test_data['label'].astype(np.int64), num_classes, 64)

    # get handle at init step
    train_label_pmf = dist_homo_graph.get_node_value_handle('train_label')
    valid_label_pmf = dist_homo_graph.get_node_value_handle('valid_label')
    test_label_pmf  = dist_homo_graph.get_node_value_handle('test_label')
    indegree_pmf = dist_homo_graph.get_node_value_handle('in_degree')
    outdegree_pmf = dist_homo_graph.get_node_value_handle('out_degree')
    handles = Handles(train_label_handle=train_label_pmf, valid_label_handle=valid_label_pmf, test_label_handle=test_label_pmf,
                      out_degree_handle=outdegree_pmf, in_degree_handle=indegree_pmf)

    main(options, dist_homo_graph, train_data, valid_data, test_data, handles)

    wg.shutdown()

