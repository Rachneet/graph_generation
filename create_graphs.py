import networkx as nx
import numpy as np

from load_data import *


# graph_type corresponds to 'cora, 'citeseer', 'pubmed'
def create(args):

    if args.graph_type == 'citeseer':

        _, _, G = graph_load(args.graph_type)
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs=[]
        for i in range(G.number_of_nodes()):
            # number of egos = number of nodes
            # person knowing other person
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)

        args.max_prev_node = 250

    elif args.graph_type == 'barabasi_density':
        graphs =[]
        for i in range(50,100):
            for j in range(4,8):
                for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i, j))

        args.max_prev_node = 87

    elif 'barabasi_small' in args.graph_type:
        graphs = []
        for i in range(100, 101):
            for j in range(4,5):
                # for k in range(10):
                #     graphs.append(nx.barabasi_albert_graph(i, j))
                graphs.append(nx.barabasi_albert_graph(i, j))
        args.max_prev_node = None

    elif args.graph_type == 'primary_school':
        graphs = []
        for i in range(1, 20):
            with open("/home/rachneet/PycharmProjects/primary_school_dataset/graph_" + str(i) + ".csv",
                      'rb') as csvfile:
                G = nx.read_edgelist(csvfile, delimiter=',',nodetype=int)
                graphs.append(G)
        args.max_prev_node = 156

    elif args.graph_type.startswith('community'):
        graphs=[]
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        # c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # c_sizes = [15] * num_communities
        for k in range(3000):
            c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
            graphs.append(n_community(c_sizes, p_inter=0.01))
        args.max_prev_node = 25

    return graphs


