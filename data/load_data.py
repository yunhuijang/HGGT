import pickle
import networkx as nx
import torch
from tqdm import tqdm
import pandas as pd

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import train_val_test_split, adj_to_k2_tree, map_new_ordered_graph, adj_to_graph, tree_to_bfs_string
from data.mol_utils import canonicalize_smiles, smiles_to_mols, add_self_loop, tree_to_bfs_string_mol, mols_to_nx


DATA_DIR = "resource"

def generate_string(dataset_name, order='C-M', k=2):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    if dataset_name == 'planar':
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{dataset_name}/{dataset_name}.pt')
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
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
        strings = [tree_to_bfs_string_mol(tree) for tree in tqdm(trees, 'Generating strings from tree')]
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