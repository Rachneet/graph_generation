import numpy as np
import networkx as nx
import pickle
import re
import load_data
import matplotlib.pyplot as plt


def citeseer_ego():
    _, _, G = load_data.Graph_load(dataset='citeseer')
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    return graphs


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G


def pick_connected_component_new(G):
    # print('in pick connected component new')
    # print(G.number_of_nodes())
    # print(type(G))
    # adj_list = G.adjacency_list()
    # print(adj_list)
    # print(len(adj_list))
    for id,adj in G.adjacency():
        # print('id : adj:', id,adj)
        if len(adj) == 0:
            id_min = 0
        else:
            id_min = min(adj)
        # print('id_min: ', id_min)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    # print(type(node_list))
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def snap_txt_output_to_nx(in_fname):
    G = nx.Graph()
    with open(in_fname, 'r') as f:
        for line in f:
            if not line[0] == '#':
                splitted = re.split('[ \t]', line)

                # self loop might be generated, but should be removed
                u = int(splitted[0])
                v = int(splitted[1])
                if not u == v:
                    G.add_edge(int(u), int(v))
    return G


# load a list of graphs
def load_graph_list(fname,is_real=True):
    # print('in load graph list')
    # print(fname)
    with open(fname, "rb") as file:
        # print("in file open")
        graph_list = pickle.load(file)
        #print(graph_list)
    for i in range(len(graph_list)):
        # print('in for')
        # print(type(graph_list[i]))
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        # print(len(edges_with_selfloops))

        if len(edges_with_selfloops)>0:
            # print('pass 1')
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            # print('is real')
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])

    return graph_list


# draw a list of graphs [G]
def draw_graph_list(G_list, row, col, fname = 'figures/test', layout='spring', is_single=False,k=1,node_size=55,alpha=1,width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    plt.switch_backend('agg')
    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        # if i%2==0:
        #     plt.title('real nodes: '+str(G.number_of_nodes()), fontsize = 4)
        # else:
        #     plt.title('pred nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # parts = community.best_partition(G)
        # values = [parts.get(node) for node in G.nodes()]
        # colors = []
        # for i in range(len(values)):
        #     if values[i] == 0:
        #         colors.append('red')
        #     if values[i] == 1:
        #         colors.append('green')
        #     if values[i] == 2:
        #         colors.append('blue')
        #     if values[i] == 3:
        #         colors.append('yellow')
        #     if values[i] == 4:
        #         colors.append('orange')
        #     if values[i] == 5:
        #         colors.append('pink')
        #     if values[i] == 6:
        #         colors.append('black')
        plt.axis("off")
        if layout=='spring':
            pos = nx.spring_layout(G,k=k/np.sqrt(G.number_of_nodes()),iterations=100)
            # pos = nx.spring_layout(G)

        elif layout=='spectral':
            pos = nx.spectral_layout(G)
        # # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
        # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2, font_size = 1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3,width=0.2)

        # plt.axis('off')
        # plt.title('Complete Graph of Odd-degree Nodes')
        # plt.show()
    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=600)
    plt.close()


def export_graphs_to_txt(g_list, output_filename_prefix):
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + '_' + str(i) + '.txt', 'w+')
        for (u, v) in G.edges():
            idx_u = G.nodes().index(u)
            idx_v = G.nodes().index(v)
            f.write(str(idx_u) + '\t' + str(idx_v) + '\n')
        i += 1


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    # adj = adj[~np.all(adj == 0, axis=1)]
    # adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def perturb(graph_list, p_del, p_add=None):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        edges = list(G.edges())
        i = 0
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1
        if p_add is None:
            num_nodes = G.number_of_nodes()
            p_add_est = np.sum(trials) / (num_nodes * (num_nodes - 1) / 2 -
                                          G.number_of_edges())
        else:
            p_add_est = p_add

        nodes = list(G.nodes())
        tmp = 0
        for i in range(len(nodes)):
            u = nodes[i]
            trials = np.random.binomial(1, p_add_est, size=G.number_of_nodes())
            j = 0
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                if trials[j] == 1:
                    tmp += 1
                    G.add_edge(u, v)
                j += 1

        perturbed_graph_list.append(G)
    return perturbed_graph_list


def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


if __name__=='__main__':
    # graphs =load_graph_list('graphs/'+ 'GraphRNN_RNN_community2_multi_4_128_train_0.dat',is_real=True)
    # graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_barabasi_small_4_64_train_0.dat')
    #graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_barabasi_small_4_64_test_0.dat')
    #graphs = load_graph_list('graphs/' + 'Baseline_DGMG_barabasi_small_64_pred_1900.dat')
    # graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_primary_school_4_128_pred_2800_1.dat')
    # graphs = load_graph_list('baselines/graphvae/graphs/' + 'barabasi_vae_new_0.dat')
    graphs =load_graph_list("/home/rachneet/PycharmProjects/nevae-master_recon/graph/nevae_community_pred.dat")
    #graphs =load_graph_list("/home/rachneet/PycharmProjects/dgl_dgmg/graph_pred_community_dgmg_new_7.dat")

    # for barabasi 170 train graphs and 1024 pred graphs
    print(len(graphs))

    for i in range(0,160,16):
        # draw_graph_list(graphs[i:i+16], 4, 4, fname='baselines/graphvae/figures_prediction/barabasi_vae_new_'+ str(i))
        draw_graph_list(graphs[i:i + 16], 4, 4, fname='./figures_nevae_community/community_multi_nevae_' + str(i))


