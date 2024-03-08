import pickle
import networkx as nx
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import graphgt
from scipy.sparse import lil_matrix, vstack

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import train_val_test_split, adj_to_k2_tree, map_new_ordered_graph, adj_to_graph, tree_to_bfs_string
from data.mol_utils import canonicalize_smiles, smiles_to_mols, add_self_loop, tree_to_bfs_string_mol, mols_to_nx


DATA_DIR = "resource"

def generate_string(dataset_name, order='C-M', k=2):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    if dataset_name in ['planar', 'sbm']:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{dataset_name}/{dataset_name}.pt')
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
        
    elif dataset_name == 'proteins':
        adjs = load_proteins_data(DATA_DIR)
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    elif dataset_name in ['profold', 'collab']:
        # adjs = np.load(f'{DATA_DIR}/METR_LA/adj.npy')
        dataloader = graphgt.DataLoader(name=dataset_name, save_path=f'./resource/', format='numpy')
        adjs = dataloader.adj[:500]
        graphs = [adj_to_graph(adj) for adj in adjs]
    elif dataset_name == 'lobster':
        graphs = []
        p1 = 0.7
        p2 = 0.7
        count = 0
        min_node = 10
        max_node = 100
        max_edge = 0
        mean_node = 80
        num_graphs = 100

        seed_tmp = 1234
        while count < num_graphs:
            G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
            if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                graphs.append(G)
                if G.number_of_edges() > max_edge:
                    max_edge = G.number_of_edges()
                count += 1
            seed_tmp += 1
    elif dataset_name == 'point':
        graphs = load_point_data(DATA_DIR, min_num_nodes=0, max_num_nodes=10000, 
                            node_attributes=False, graph_labels=True)
    elif dataset_name == 'ego':
        _, _, G = load_ego_data(dataset='citeseer')
        G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
    else:
        with open (f'{DATA_DIR}/{dataset_name}/{dataset_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, dataset_name)
    graph_list = []
    for graphs in train_graphs, val_graphs, test_graphs:
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, seed=0, is_mol=True)
        new_ordered_graphs = [map_new_ordered_graph(graph) for graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    # write graphs
    splits = ['train', 'val', 'test']
    
    for graphs, split in zip(graph_list, splits):
        adjs = [nx.adjacency_matrix(graph, range(len(graph))) for graph in graphs]
        trees = [adj_to_k2_tree(torch.Tensor(adj.todense()), return_tree=True, k=k, is_mol=False) for adj in tqdm(adjs, 'Generating tree from adj')]
        strings = [tree_to_bfs_string(tree, string_type='group') for tree in tqdm(trees, 'Generating strings from tree')]
        file_name = f'{dataset_name}_str_{split}'
        with open(f'{DATA_DIR}/{dataset_name}/{order}/{file_name}_{k}.txt', 'w') as f:
            for string in strings:
                f.write(f'{string}\n')
        if split == 'test':
            with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_test_graphs.pkl', 'wb') as f:
                pickle.dump(graphs, f)
    return graph_list
                
def generate_mol_string(dataset_name, order='C-M', is_small=False):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    col_dict = {'qm9': 'SMILES1', 'zinc': 'smiles'}
    df = pd.read_csv(f'{DATA_DIR}/{dataset_name}/{dataset_name}.csv')
    smiles = list(df[col_dict[dataset_name]])
    if is_small:
        smiles = smiles[:100]
    smiles = [s for s in smiles if len(s)>1]
    smiles = canonicalize_smiles(smiles)
    splits = ['train', 'val', 'test']
    train_smiles, val_smiles, test_smiles = train_val_test_split(smiles, dataset_name)
    for s, split in zip([train_smiles, val_smiles, test_smiles], splits):
        if is_small:
            with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_small_smiles_{split}.txt', 'w') as f:
                for string in s:
                    f.write(f'{string}\n')
        else:
            with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_smiles_{split}.txt', 'w') as f:
                for string in s:
                    f.write(f'{string}\n')
    graph_list = []
    for smiles in train_smiles, val_smiles, test_smiles:
        mols = smiles_to_mols(smiles)
        graphs = mols_to_nx(mols)
        graphs = [add_self_loop(graph) for graph in tqdm(graphs, 'Adding self-loops')]
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_graphs = graphs
        total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0, is_mol=True)
        new_ordered_graphs = [map_new_ordered_graph(graph) for graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    # write graphs
    
    for graphs, split in zip(graph_list, splits):
        weighted_adjs = [nx.attr_matrix(graph, edge_attr='label', rc_order=range(len(graph))) for graph in graphs]
        trees = [adj_to_k2_tree(torch.Tensor(adj), return_tree=True, is_mol=True) for adj in tqdm(weighted_adjs, 'Generating tree from adj')]
        strings = [tree_to_bfs_string_mol(tree, string_type='group') for tree in tqdm(trees, 'Generating strings from tree')]
        if is_small:
            file_name = f'{dataset_name}_small_str_{split}'
        else:
            file_name = f'{dataset_name}_str_{split}'
        with open(f'{DATA_DIR}/{dataset_name}/{order}/{file_name}.txt', 'w') as f:
            for string in strings:
                f.write(f'{string}\n')
        if split == 'test':
            if is_small:
                with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_small_test_graphs.pkl', 'wb') as f:
                    pickle.dump(graphs, f)
            else:
                with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_test_graphs.pkl', 'wb') as f:
                    pickle.dump(graphs, f)
                
# codes adapted from https://github.com/KarolisMart/SPECTRE
def load_proteins_data(data_dir):
    
    min_num_nodes=100
    max_num_nodes=500
    
    adjs = []
    eigvals = []
    eigvecs = []
    n_nodes = []
    n_max = 0
    max_eigval = 0
    min_eigval = 0

    G = nx.Graph()
    # Load data
    path = os.path.join(data_dir, 'proteins/DD')
    data_adj = np.loadtxt(os.path.join(path, 'DD_A.txt'), delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)
    data_graph_types = np.loadtxt(os.path.join(path, 'DD_graph_labels.txt'), delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges
    G.add_edges_from(data_tuple)
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    for i in tqdm(range(graph_num)):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_types[i]
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            adj = torch.from_numpy(nx.adjacency_matrix(G_sub).toarray()).float()
            L = nx.normalized_laplacian_matrix(G_sub).toarray()
            L = torch.from_numpy(L).float()
            eigval, eigvec = torch.linalg.eigh(L)
            
            eigvals.append(eigval)
            eigvecs.append(eigvec)
            adjs.append(adj)
            n_nodes.append(G_sub.number_of_nodes())
            if G_sub.number_of_nodes() > n_max:
                n_max = G_sub.number_of_nodes()
            max_eigval = torch.max(eigval)
            if max_eigval > max_eigval:
                max_eigval = max_eigval
            min_eigval = torch.min(eigval)
            if min_eigval < min_eigval:
                min_eigval = min_eigval

    return adjs

def load_point_data(data_dir, min_num_nodes, max_num_nodes, node_attributes, graph_labels):
    print('Loading point cloud dataset')
    name = 'FIRSTMM_DB'
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, f'{name}_A.txt'), delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(os.path.join(path, f'{name}_node_attributes.txt'), 
                                   delimiter=',')
    data_node_label = np.loadtxt(os.path.join(path, f'{name}_node_labels.txt'), 
                                 delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, f'{name}_graph_indicator.txt'),
                                      delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(os.path.join(path, f'{name}_graph_labels.txt'), 
                                       delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    
    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
            G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]

        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
        if G_sub.number_of_nodes() > max_nodes:
            max_nodes = G_sub.number_of_nodes()
            
    print('Loaded')
    return graphs

# Codes adpated from https://github.com/JiaxuanYou/graph-generation
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_ego_data(dataset):
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open(f"{DATA_DIR}/ego/ind.{dataset}.{names[i]}", 'rb'), encoding='latin1')
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{DATA_DIR}/ego/ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended

    features = vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G