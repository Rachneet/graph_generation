import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from random import *
from utils import *
import args


# load ENZYMES and PROTEIN and DD dataset
def graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def test_graph_load_DD():
    graphs, max_num_nodes = graph_load_batch(min_num_nodes=10,name='DD',node_attributes=False,graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0:row*col], row=row,col=col, fname='figures/test')
    print('max num nodes',max_num_nodes)


# index file parser
def parse_index_file(filename):
    index=[]
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# load citeseer, cora and pubmed
def graph_load(dataset='cora'):

    names= ['x', 'tx', 'allx', 'graph']
    objects= []

    for i in range(len(names)):
        load = pkl.load(open('dataset/ind.{}.{}'.format(dataset,names[i]), 'rb'), encoding='latin1')
        objects.append(load)

    x,tx,allx,graph = tuple(objects)
    print(x.shape, tx.shape, allx.shape)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    print(G.number_of_edges(), G.number_of_nodes())
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_adj(adj, max_prev_node=10, is_full = False):
    '''
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


# using pytorch dataloader
class graph_sequence_sampler(Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes)

        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node

        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # take random permutation
        x_idx = np.random.permutation(adj_copy.shape[0])
        # array([18, 11, 19,  2,  8,  0, 17, 12,  1,  3, 15, 10,  5, 13, 16, 14,  4,
        #         7,  9,  6])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)] # shuffled adjacency
        adj_copy_matrix = np.asmatrix(adj_copy) # shuffled adj matrix
        G = nx.from_numpy_matrix(adj_copy_matrix) # new graph w/ shuffled nodes
        if adj_copy.shape[0] != 0:
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            # eg. output [ 5  0  1  2  3  4  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[0:adj_encoded.shape[0], :] = adj_encoded
            x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # print("shape:", adj_copy.shape[0])
            if adj_copy.shape[0] != 0:
                # then do bfs in the permuted G
                start_idx = np.random.randint(adj_copy.shape[0])
                x_idx = np.array(bfs_seq(G, start_idx))
                adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
                # print(adj_copy)
                # encode adj
                adj_encoded = encode_adj_flexible(adj_copy.copy())
                # print("Encoded adj:",adj_encoded)

                max_encoded_len = [len(adj_encoded[i]) for i in range(len(adj_encoded))]
                if len(max_encoded_len) != 0:
                    max_encoded_len = max(max_encoded_len)
                    max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



if __name__ == '__main__' :
    graph_load()
    adj, feats, G = graph_load()
    print(type(adj), type(feats), type(G))
    print(adj.shape, feats.shape)

