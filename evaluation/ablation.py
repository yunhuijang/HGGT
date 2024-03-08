import os
from pathlib import Path
import torch
import pickle
import networkx as nx
from statistics import mean
import pandas as pd

from data.data_utils import remove_redundant
from data.load_data import load_proteins_data
from data.orderings import bw_from_adj, ORDER_FUNCS, order_graphs


DATA_DIR = "resource"

def compute_compression_rate(data_name, order, model, is_red=False):
    raw_dir = f"{DATA_DIR}/{data_name}/{order}"
    total_strings = []
    # load k2 tree string
    for split in ['train', 'val', 'test']:
        string_path = os.path.join(raw_dir, f"{data_name}_str_{split}.txt")
        strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        # for string length
        if is_red:
            strings = [''.join(remove_redundant(string)) for string in strings]
        
        total_strings.extend(strings)

    
    # load data
    if data_name in ['planar', 'sbm']:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{data_name}/{data_name}.pt')
        adjs = [adj.numpy() for adj in adjs]
    elif data_name == 'proteins':
        adjs = load_proteins_data("../resource")
    else:
        with open (f'{DATA_DIR}/{data_name}/{data_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
        
        order_func = ORDER_FUNCS['BFS']
        ordered_graphs = order_graphs(graphs, num_repetitions=1, order_func=order_func, is_mol=False, seed=0)
        ordered_graphs = [nx.from_numpy_array(ordered_graph.to_adjacency().numpy()) for ordered_graph in ordered_graphs]
        adjs = [nx.adjacency_matrix(graph).toarray() for graph in ordered_graphs]
        
    n_nodes = [adj.shape[0]*adj.shape[1] for adj in adjs]
    if model == 'hggt':
        pass
    # len_strings = [len(string) for string in total_strings]
        
    else:
        # GraphRNN
        len_strings = []
        n_squares = []
        for adj in adjs:
            b = bw_from_adj(adj)
            n = adj.shape[0]
            len_strings.append(n*b-((b*b+b)/2))
            n_squares.append(n*n)
            
    compression_rates = [length / n_square for length, n_square in zip(len_strings, n_squares)]
    
    return mean(compression_rates)


datas = ['proteins']
orders = ['BFS', 'DFS', 'C-M']
result_df = pd.DataFrame(columns = datas, index=orders)

def get_max_len(data_name, order='C-M', k=2):
    total_strings = []
    k_square = k**2
    for split in ['train', 'test', 'val']:
        if k > 2:
            string_path = os.path.join(DATA_DIR, f"{data_name}/{order}/{data_name}_str_{split}_{k}.txt")
        else:
            string_path = os.path.join(DATA_DIR, f"{data_name}/{order}/{data_name}_str_{split}.txt")
        
        # string_path = os.path.join(DATA_DIR, f"{data_name}/{order}/{data_name}_str_{split}_{k}.txt")
        strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        
        total_strings.extend(strings)

    # red_strings = [''.join(red) for red in red_list]
    
    max_len = max([len(string) for string in total_strings])
    group_max_len = max_len / k_square
    red_len = [len(remove_redundant(string)) for string in total_strings]
    
    return max_len, group_max_len, red_len

for data in ['GDSS_com', 'planar', 'GDSS_enz', 'GDSS_grid']:
    # rnn_string = compute_compression_rate(data, 'BFS', 'graphrnn', True)
    _, rnn_string, red_len = get_max_len(data)
    print(round(mean(red_len), 3))


# result_df.to_csv('compression.csv')


