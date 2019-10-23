from train import *
import os
from args import *
import create_graphs
import random as rand
import time
from load_data import *

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    # if args.clean_tensorboard:
    #     if os.path.isdir("tensorboard"):
    #         shutil.rmtree("tensorboard")
    # configure("tensorboard/run" + time, flush_secs=5)

    graphs = create_graphs.create(args)
    # # save_graph_list(graphs, args.graph_save_path + args.fname + '0.dat')
    # # # split datasets
    rand.seed(2345)
    rand.shuffle(graphs)
    graphs_len = len(graphs)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[0:int(0.2 * graphs_len)]
    graphs_train = graphs

    # modification
    # save_graph_list(graphs_train, args.graph_save_path + args.fname_train + '0.dat')
    # save_graph_list(graphs_test, args.graph_save_path + args.fname_test + '0.dat')
    # save_graph_list(graphs_validate, args.graph_save_path + args.fname_validate + '0.dat')

    # if use pre-saved graphs
    # dir_input = "./graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_len = len(graphs)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

    # if use pre-saved graphs : modification
    # dir_input = "./graphs/"
    # fname_train = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_train_' + str(0) + '.dat'
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # fname_validate = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs_train = load_graph_list(fname_train, is_real=True)
    # graphs_test = load_graph_list(fname_test, is_real=True)
    # graphs_validate = load_graph_list(fname_validate, is_real=True)

    # graph_validate_len = 0
    # for graph in graphs_validate:
    #     graph_validate_len += graph.number_of_nodes()
    # graph_validate_len /= len(graphs_validate)
    # print('graph_validate_len', graph_validate_len)
    #
    # graph_test_len = 0
    # for graph in graphs_test:
    #     graph_test_len += graph.number_of_nodes()
    # graph_test_len /= len(graphs_test)
    # print('graph_test_len', graph_test_len)

    args.max_num_node = max([graphs_train[i].number_of_nodes() for i in range(len(graphs_train))])
    max_num_edge = max([graphs_train[i].number_of_edges() for i in range(len(graphs_train))])
    min_num_edge = min([graphs_train[i].number_of_edges() for i in range(len(graphs_train))])

    # args.max_num_node = 2000
    # show graphs statistics
    # print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    # To get train and test set, after loading you need to manually slice
    # save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    # save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    # print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ####  modification
    # save_graph_list(graphs_train, args.graph_save_path + args.fname_train + '0.dat')
    # save_graph_list(graphs_test, args.graph_save_path + args.fname_test + '0.dat')
    # save_graph_list(graphs_validate, args.graph_save_path + args.fname_validate + '0.dat')
    # print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ## comment when normal training, for graph completion only
    # p = 0.8
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
    #     for edge in list(graph.edges()):
    #         # print('edge',edge)
    #         if np.random.rand()>p:
    #             graph.remove_edge(edge[0],edge[1])

    def get_edge_list(graph, node):
        edge_list = []
        for u, v in list(graph.edges()):
            # print(u,v)
            if node == u or node == v:
                # print(u,v)
                edge_list.append((u, v))
        return edge_list

    # randomly remove p amount of nodes
    p = 0.2
    for graph in graphs:
        count = 0
        edge_list = []
        num_nodes = len(graph.nodes())
        num_remove = round(p * num_nodes)
        print("to remove:", num_remove)
        for i in range(num_remove):
            node_to_remove = rand.choice(list(graph.nodes()))
            edge_list = get_edge_list(graph, node_to_remove)
            graph.remove_nodes_from([node_to_remove])
            if nx.is_connected(graph):
                count += 1
                continue
            else:
                graph.add_nodes_from([node_to_remove])
                graph.add_edges_from(edge_list)
                num_remove += 1
    print("removed:", count)

    args.nodes_removed = count
    # for graph in graphs_train:
    #     if len(list(graph.nodes())) == 0:
    #         graphs_train.remove(graph)


    ### dataset initialization
    # if 'nobfs' in args.note:
    #     print('nobfs')
    #     dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
    #     args.max_prev_node = args.max_num_node - 1
    # if 'barabasi_noise' in args.graph_type:
    #     print('barabasi_noise')
    #     dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train, max_prev_node=args.max_prev_node)
    #     args.max_prev_node = args.max_num_node - 1
    # else:
    dataset = graph_sequence_sampler(graphs_train, max_prev_node=args.max_prev_node,
                                                 max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    # use when calculating the max prev node
    args.max_prev_node = dataset.max_prev_node

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    # if 'GraphRNN_VAE_conditional' in args.note:
    #     rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
    #                     hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
    #                     has_output=False).cuda()
    #     output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
    #                                        y_size=args.max_prev_node).cuda()
    # elif 'GraphRNN_MLP' in args.note:
    #     rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
    #                     hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
    #                     has_output=False).cuda()
    #     output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
    #                        y_size=args.max_prev_node).cuda()
    if 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False, output_size=args.hidden_size_rnn_output).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                           y_size=args.max_prev_node).cuda()

    ### start training
    # train(args, dataset_loader, rnn, output)

    ## train graph completion
    train_graph_completion(args, dataset_loader, rnn, output)

    # nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter=200, graph_validate_len=graph_validate_len,
    #           graph_test_len=graph_test_len)

