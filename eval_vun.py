import torch
import pickle
import networkx as nx

from evaluation.evaluation_spectre import eval_fraction_isomorphic, eval_fraction_unique_non_isomorphic_valid, eval_acc_grid_graph, eval_acc_planar_graph, eval_acc_sbm_graph, is_grid_graph, is_planar_graph, is_sbm_graph, eval_fraction_unique
from evaluation.evaluation_spectre import eval_fraction_unique
from data.data_utils import train_val_test_split, adj_to_graph
from data.load_data import load_proteins_data

gcg_dict = {'GDSS_com': 'May02-00:49:13', 'GDSS_grid': 'May02-15:56:32', 'GDSS_ego': "May14-07:56:26",
            'GDSS_enz': 'May02-06:39:38' , 'planar': "May14-11:34:11", 'sbm': 'Jun23-07:39:09'}

graphgen_dict = {'GDSS_com': 'DFScodeRNN_com_small_2023-05-14 01:40:56', 
                 'GDSS_grid': 'DFScodeRNN_grid_2023-05-14 01:43:04', 
                 'GDSS_ego': "DFScodeRNN_ego_small_2023-05-14 01:39:50",
                 'GDSS_enz': 'DFScodeRNN_enz_2023-05-14 02:11:21',
                 'planar': "DFScodeRNN_planar_2023-05-14 01:53:01"}

digress_dict = {'GDSS_com': '2023-05-13/14-54-28', 
                 'GDSS_grid': '2023-05-13/15-02-58', 
                 'GDSS_ego': '2023-05-13/14-53-38',
                 'GDSS_enz': '2023-05-13/14-56-45',
                 'planar': "2023-05-13/13-49-30"}

DATA_DIR = "resource"

def load_generated_graphs(data_name, method):
    if method == 'train':
        # for train
        with open(f'resource/{data_name}/C-M/{data_name}_test_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'digress':
        # digress
        file_name = digress_dict[data_name]
        with open(f'../DiGress/src/outputs/{file_name}/graphs/generated_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
        adjs = [graph[1] for graph in graphs]
        graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
        
    elif method == 'graphgen':
        file_name = graphgen_dict[data_name]
        with open(f'../graphgen/graphs/{file_name}/generated_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'gcg':
        file_name = gcg_dict[data_name]
        with open(f'samples/graphs/{data_name}/{file_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'gdss':
        data_dict = {'GDSS_com': 'community_small', 'GDSS_enz': 'ENZYMES', 'GDSS_grid': 'grid'}
        with open(f'../GDSS/samples/pkl/{data_dict[data_name]}/test/{data_name}_sample.pkl', 'rb') as f:
            graphs = pickle.load(f)
    
    return graphs

def load_train_graphs(data_name):
    if data_name in ['planar', 'sbm']:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{data_name}/{data_name}.pt')
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
        
    elif data_name == 'proteins':
        adjs = load_proteins_data(DATA_DIR)
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    else:
        with open (f'{DATA_DIR}/{data_name}/{data_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, data_name)
    return train_graphs

data_name = 'planar'
gen_graphs = load_generated_graphs(data_name, 'gcg')

print(eval_fraction_unique(gen_graphs))

train_graphs = load_train_graphs(data_name)

if data_name == 'GDSS_grid':
    val = eval_acc_grid_graph(gen_graphs)
    validity_func = is_grid_graph
elif data_name == 'sbm':
    acc = eval_acc_sbm_graph(gen_graphs, refinement_steps=1000, strict=False)
    validity_func = is_sbm_graph
elif data_name == 'planar':
    acc = eval_acc_planar_graph(gen_graphs)
    validity_func = is_planar_graph
else:
    validity_func = lambda x: True

n = len(gen_graphs)
val = len([is_grid_graph(graph) for graph in gen_graphs])/n
unique, un, vun = eval_fraction_unique_non_isomorphic_valid(gen_graphs, train_graphs, validity_func=validity_func)
novel = eval_fraction_isomorphic(gen_graphs, train_graphs)

print(f'validity: {round(val, 3)}')
print(f'unique: {round(unique, 3)}')
print(f'novel: {round(1 - novel, 3)}')
if data_name == 'proteins':
    print(f'un: {round(un, 3)}')
else:
    print(f'vun: {round(vun, 3)}')
