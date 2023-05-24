import torch
from torch.nn import ZeroPad2d
from torch import LongTensor
from torch.utils.data import random_split
from torch import count_nonzero
import math
from collections import deque
from treelib import Tree, Node
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import networkx as nx
from treelib import Tree, Node
import numpy as np
from itertools import compress, islice
import os
from pathlib import Path
import json


from data.tokens import grouper_mol


DATA_DIR = "gcg/resource"
NODE_TYPE_DICT = {'F': 9, 'O': 10, 'N': 11, 'C': 12, 'P': 13, 'I': 14, 'Cl': 15, 'Br': 16, 'S': 17}
TYPE_NODE_DICT = {str(key): value for value, key in NODE_TYPE_DICT.items()}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
TYPE_BOND_DICT = {key: value for value, key in NODE_TYPE_DICT.items()}

def get_level(node):
    return int(node.identifier.split('-')[0])

def get_location(node):
    return int(node.identifier.split('-')[1])

def get_k(tree):
    return math.sqrt(len(tree[tree.root].successors(tree.identifier)))

def get_parent(node, tree):
    return tree[node.predecessor(tree.identifier)]

def get_children_identifier(node, tree):
    return sorted(node.successors(tree.identifier), key=lambda x: get_sort_key(x))

def get_sort_key(node_id):
    if len(node_id.split('-')) > 2:
        return (int(node_id.split('-')[0]), int(node_id.split('-')[1]), int(node_id.split('-')[2]))
    else:
        return (int(node_id.split('-')[0]), int(node_id.split('-')[1]))
    
def nearest_power(N, base=2):
    a = int(math.log(N, base)) 
    if base**a == N:
        return N

    return base**(a + 1)

def adj_to_k2_tree(adj, return_tree=False, is_wholetree=False, k=4, is_mol=False):
    if not is_mol:
        adj[adj > 0] = 1
    n_org_nodes = adj.shape[0]
    # add padding (proper size for k)
    n_nodes = nearest_power(n_org_nodes, k)
    k_square = k**2
    padder = ZeroPad2d((0, n_nodes-n_org_nodes, 0, n_nodes-n_org_nodes))
    padded_adj = padder(adj)
    total_level = int(math.log(n_nodes, k))
    tree_list = []
    leaf_list = []
    tree = Tree()
    # add root node
    tree.create_node("root", "0")
    tree_key_list = deque([])
    slice_size = int(n_nodes / k)
    # slice matrices 
    start_index = range(0,n_nodes,slice_size)
    end_index = range(slice_size,n_nodes+1,slice_size)
    slices = []
    for row_start, row_end in zip(start_index, end_index):
        for col_start, col_end in zip(start_index, end_index):
            slices.append(padded_adj[row_start:row_end, col_start:col_end])
    
    # slices = [padded_adj[:slice_size, :slice_size], padded_adj[:slice_size, slice_size:], padded_adj[slice_size:, :slice_size], padded_adj[slice_size:, slice_size:]]
    sliced_adjs = deque(slices)
    sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in sliced_adjs])
    tree_list.append(sliced_adjs_is_zero)
    # molecule + only leaf
    if is_mol and adj.shape[0] == k:
        tree_element_list = deque(list(map(int, torch.flatten(adj).tolist())))
    else:
        tree_element_list = deque(sliced_adjs_is_zero)

    for i, elem in enumerate(tree_element_list, 1):
        tree.create_node(elem, f"1-{i}", parent="0")
        tree_key_list.append(f"1-{i}")
    
    while (slice_size != 1):
        n_nodes = sliced_adjs[0].shape[0]
        if n_nodes == k:
            if is_wholetree:
                leaf_list = [adj.reshape(k_square,) for adj in sliced_adjs]
            else:
                leaf_list = [adj.reshape(k_square,) for adj in sliced_adjs if count_nonzero(adj)>0]
            break
        slice_size = int(n_nodes / k)
        target_adj = sliced_adjs.popleft()
        target_adj_size = target_adj.shape[0]
        if return_tree:
            parent_node_key = tree_key_list.popleft()
        # remove adding leaves to 0
        if not is_wholetree:
            if count_nonzero(target_adj) == 0:
                continue
        # generate tree_list and leaf_list
        new_sliced_adjs = []
        start_index = range(0,n_nodes,slice_size)
        end_index = range(slice_size,n_nodes+1,slice_size)
        for row_start, row_end in zip(start_index, end_index):
            for col_start, col_end in zip(start_index, end_index):
                new_sliced_adjs.append(target_adj[row_start:row_end, col_start:col_end])
        # new_sliced_adjs = [target_adj[:slice_size, :slice_size], target_adj[:slice_size, slice_size:], 
        #         target_adj[slice_size:, :slice_size], target_adj[slice_size:, slice_size:]]
        new_sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in new_sliced_adjs])
        sliced_adjs.extend(new_sliced_adjs)
        tree_list.append(new_sliced_adjs_is_zero)
        
        if return_tree:
            # generate tree
            tree_element_list.extend(new_sliced_adjs_is_zero)
            cur_level = int(total_level - math.log(target_adj_size, k) + 1)
            cur_level_key_list = [int(key.split('-')[1]) for key in tree_key_list if int(key.split('-')[0]) == cur_level]
            if len(cur_level_key_list) > 0:
                key_starting_point = max(cur_level_key_list)
            else:
                key_starting_point = 0
            for i, elem in enumerate(new_sliced_adjs_is_zero, key_starting_point+1):
                tree.create_node(elem, f"{cur_level}-{i}", parent=parent_node_key)
                tree_key_list.append(f"{cur_level}-{i}")
            
    if return_tree:
        # add leaves to tree
        leaves = [node for node in tree.leaves() if node.tag == 1]
        index = 1
        for leaf, leaf_values in zip(leaves, leaf_list):
            for value in leaf_values:
                tree.create_node(int(value), f"{total_level}-{index}", parent=leaf)
                index += 1
        return tree
    else:
        return tree_list, leaf_list

def check_tree_validity(tree):
    depth = tree.depth()
    if depth == 1:
        return False
    leaves = [leaf for leaf in tree.leaves() if leaf.tag != '0']
    invalid_leaves = [leaf for leaf in leaves if tree.depth(leaf)!=depth]
    if len(invalid_leaves) == 0:
        return True
    else:
        return False

def tree_to_adj(tree, k=2):
    '''
    convert k2 tree to adjacency matrix
    '''
    tree = map_starting_point(tree, k)
    depth = tree.depth()
    leaves = [leaf for leaf in tree.leaves() if leaf.tag != '0']
    one_data_points = [leaf.data for leaf in leaves]
    x_list = [data[0] for data in one_data_points]
    y_list = [data[1] for data in one_data_points]
    label_list = [NODE_TYPE_DICT[leaf.tag] if leaf.tag in NODE_TYPE_DICT.keys() else int(leaf.tag) for leaf in leaves]
    matrix_size = int(k**depth)
    adj = torch.zeros((matrix_size, matrix_size))
    for x, y, label in zip(x_list, y_list, label_list):
        if (x > len(adj)) or (y > len(adj)):
            return None
        adj[x, y] = label
    
    return adj

def map_starting_point(tree, k):
    '''
    map starting points for each elements in tree (to convert adjacency matrix)
    '''
    try:
        bfs_list = [tree[node] for node in tree.expand_tree(mode=Tree.WIDTH, 
                                                            key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1]), int(x.identifier.split('-')[2])))]
    except:
        bfs_list = [tree[node] for node in tree.expand_tree(mode=Tree.WIDTH, 
                                                            key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))]
    bfs_list[0].data = (0,0)
    
    for node in bfs_list[1:]:
        parent = get_parent(node, tree)
        siblings = get_children_identifier(parent, tree)
        index = siblings.index(node.identifier)
        level = get_level(node)
        tree_depth = tree.depth()
        matrix_size = k**tree_depth
        adding_value = int(matrix_size/(k**level))
        parent_starting_point = parent.data
        node.data = (parent_starting_point[0]+adding_value*int(index/k), parent_starting_point[1]+adding_value*int(index%k))
            
    return tree

def map_child_deg(node, tree):
    '''
    return sum of direct children nodes' degree (tag)
    '''
    if node.is_leaf():
        return str(int(node.tag))
    
    children = get_children_identifier(node, tree)
    child_deg = sum([int(tree[child].tag > 0) for child in children])
    
    return str(child_deg)

def map_all_child_deg(node, tree):
    '''
    return sum of all children nodes' degree (tag)
    '''
    if node.is_leaf():
        return str(int(node.tag))
    
    children = get_children_identifier(node, tree)
    child_deg = sum([int(tree[child].tag) for child in children])
    
    return str(child_deg)

def tree_to_bfs_string(tree, string_type='bfs'):
    bfs_node_list = [tree[node] for node in tree.expand_tree(mode=tree.WIDTH,
                                                             key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))][1:]
    if string_type in ['bfs', 'group', 'bfs-tri', 'group-red', 'group-red-3']:
        bfs_value_list = [str(int(node.tag)) for node in bfs_node_list]
    elif string_type in ['bfs-deg', 'bfs-deg-group']:
        bfs_value_list = [map_child_deg(node, tree) for node in bfs_node_list]
    
    return ''.join(bfs_value_list)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def bfs_string_to_tree(string, is_zinc=False, k=2):
    k_square = k**2
    tree = Tree()
    tree.create_node("root", "0")
    parent_node = tree["0"]
    node_deque = deque([])
    if is_zinc:
        node_groups = grouper_mol(string)
    else:
        node_groups = grouper(k_square, string)
    for node_1, node_2, node_3, node_4 in node_groups:
        parent_level = get_level(parent_node)
        cur_level_max = max([get_location(node) for node in tree.nodes.values() if get_level(node) == parent_level+1], default=0)
        for i, node_tag in enumerate([node_1, node_2, node_3, node_4], 1):
            if node_tag == None:
                break
            new_node = Node(tag=node_tag, identifier=f"{parent_level+1}-{cur_level_max+i}")
            tree.add_node(new_node, parent=parent_node)
            node_deque.append(new_node)
        parent_node = node_deque.popleft()
        while(parent_node.tag == '0'):
            if len(node_deque) == 0:
                return tree
            parent_node = node_deque.popleft()
    return tree

def clean_string(string):

    if "[pad]" in string:
        string = string[:string.index("[pad]")]
        
    return string

    
def train_val_test_split(
    data: list,
    data_name='GDSS_com',
    train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
    seed: int = 42,
):
    if data_name in ['qm9', 'zinc']:
        # code adpated from https://github.com/harryjo97/GDSS
        with open(os.path.join(DATA_DIR, f'{data_name}/valid_idx_{data_name}.json')) as f:
            test_idx = json.load(f)
        if data_name == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        train_idx = [i for i in range(len(data)) if i not in test_idx]
        test = [data[i] for i in test_idx]
        train_val = [data[i] for i in train_idx]
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed, shuffle=True)
    elif data_name in ['planar', 'sbm', 'proteins']:
        # code adapted from https://github.com/KarolisMart/SPECTRE
        test_len = int(round(len(data)*0.2))
        train_len = int(round((len(data) - test_len)*0.8))
        val_len = len(data) - train_len - test_len
        train, val, test = random_split(data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))
    else:
        train_val, test = train_test_split(data, train_size=train_size + val_size, shuffle=False)
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed, shuffle=True)
    return train, val, test

def adj_to_graph(adj, is_cuda=False):
    if is_cuda:
        adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 1:
        G.add_node(1)
    return G

def map_tree_pe(tree):
    depth = tree.depth()
    k = get_k(tree)
    size = int(depth*(k**2))
    pe = torch.zeros((size))
    for node in tree.nodes.values():
        node.data = pe
        if not node.is_root():
            parent = get_parent(node, tree)
            branch = get_children_identifier(parent, tree).index(node.identifier)
            current_pe = torch.zeros(int(k**2))
            current_pe[branch] = 1
            pe = torch.cat((current_pe, parent.data[:int(size-k**2)]))
            node.data = pe
    return tree
                
def map_new_ordered_graph(ordered_graph):
    '''
    Map ordered_graph object to ordered networkx graph
    '''
    org_graph = ordered_graph.graph
    ordering = ordered_graph.ordering
    mapping = {i: ordering.index(i) for i in range(len(ordering))}
    new_graph = nx.relabel_nodes(org_graph, mapping)
    return new_graph

# for redundant removed strings

def generate_final_tree_red(tree, k=2):
    tree_with_iden = add_zero_to_identifier(tree)
    final_tree = add_symmetry_to_tree(tree_with_iden, k)
    
    return final_tree

def generate_initial_tree_red(string_token_list, k=2):
    node_groups = [tuple(grouper_mol(string, k)[0]) for string in string_token_list]
    tree = Tree()
    tree.create_node("root", "0-0-0")
    parent_node = tree["0-0-0"]
    node_deque = deque([])
    for nodes in node_groups:
        parent_level = get_level(parent_node)
        cur_level_max = max([get_location(node) for node in tree.nodes.values() if get_level(node) == parent_level+1], default=0)
        for i, node_tag in enumerate(nodes, 1):
            if node_tag == None:
                break
            new_node = Node(tag=node_tag, identifier=f"{parent_level+1}-{cur_level_max+i}")
            tree.add_node(new_node, parent=parent_node)
            node_deque.append(new_node)
        parent_node = node_deque.popleft()
        while(parent_node.tag == '0'):
            if len(node_deque) == 0:
                return tree
            parent_node = node_deque.popleft()
    
    return tree

def find_new_identifier(node_id, index, is_dup=1):
    split = node_id.split('-')
    # k = 2
    if index == 1:
        num = 1
        num *= is_dup
        cur_pre_identifier = split[0] + '-' + split[1]
    # k = 3
    elif index == 2:
        num = 2
        num *= is_dup
        cur_pre_identifier = split[0] + '-' + split[1]
    elif index == 4:
        num = 1
        num *= is_dup
        cur_pre_identifier = split[0] + '-' + str(int(split[1])-2)
    elif index == 5:
        num = 1
        num *= is_dup
        cur_pre_identifier = split[0] + '-' + str(int(split[1])-1)
    else:
        num = 2
        num *= is_dup
        cur_pre_identifier = split[0] + '-' + split[1]
    
    new_last_identifier = int(split[2])-num
    return cur_pre_identifier + '-' + str(new_last_identifier)

def get_child_index(k):
    indices = []
    c = 0
    for i in range(k+1):
        for j in range(1,i+1):
            c=c+1
            if j != i:
                indices.append(c)
    return indices
    
    
def add_symmetry_to_tree(tree, k):
    k_square = k**2
    bfs_node_list = [tree[node] for node in tree.expand_tree(mode=tree.WIDTH, key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))]
    node_list = [node for node in bfs_node_list[::-1] if not node.is_leaf()]
    for node in node_list:
        child_nodes = get_children_identifier(node, tree)
        if len(child_nodes) < k_square:
            postfixes = get_child_index(k)
	
            for index in postfixes:
                copy_node = tree.get_node(child_nodes[int(index)-1])
                new_node = Node(tag=copy_node.tag, identifier=find_new_identifier(copy_node.identifier, index))
                subtree = Tree(tree.subtree(child_nodes[int(index)-1]), deep=True)
                new_tree = Tree(subtree, deep=True)
                if len(subtree) > 1:
                    for nid, n in sorted(subtree.nodes.items(), key=lambda x: (int(x[0].split('-')[0]), int(x[0].split('-')[1]), int(x[0].split('-')[2]))):
                        count_dup = len([key for key in subtree.nodes.keys() 
                                        if (key.split('-')[0] == nid.split('-')[0]) and (key.split('-')[1] == nid.split('-')[1])])
                        org_dup = len([key for key in tree.nodes.keys() 
                                        if (key.split('-')[0] == nid.split('-')[0]) and (key.split('-')[1] == nid.split('-')[1])])
                        new_iden = find_new_identifier(nid, index, (count_dup+org_dup)*100)
                        while (new_iden in tree):
                            new = int(new_iden.split('-')[2]) - 1
                            new_iden = new_iden.split('-')[0] + '-' + new_iden.split('-')[1] + '-' +  str(new)
                        new_tree.update_node(nid, identifier=new_iden)
                    tree.paste(node.identifier, new_tree)
                    
                else:
                    tree.add_node(new_node, parent=node)
            
    return tree

def add_zero_to_identifier(tree):
    new_tree = Tree(tree, deep=True)
    for node in tree.nodes:
        new_identifier = node
        while (len(new_identifier.split('-'))<3):
            new_identifier = new_identifier + '-10000'
        new_tree.update_node(node, identifier=new_identifier)
    return new_tree

def fix_symmetry(adj):
    sym_adj = torch.tril(adj) + torch.tril(adj).T
    return torch.where(sym_adj>0, 1, 0)


def map_deg_string(string):
    new_string = []
    group_queue = deque(grouper(4, string))
    group_queue.popleft()
    for index, char in enumerate(string):
        if len(group_queue) == 0:
            left = string[index:]
            break
        if char == '0':
            new_string.append(char)
        else:
            new_string.append(str(sum([int(char) for char in group_queue.popleft()])))
            
    return ''.join(new_string) + left

def remove_redundant(input_string, is_mol=False, k=2):
    k_square = k**2
    string = input_string[0:k_square]
    pos_list = list(range(1, k_square+1))
    str_pos_queue = deque([(s, p) for s, p in zip(string, pos_list)])
    if is_mol:
        group_list = list(grouper_mol(input_string))
    else:
        group_list = list(grouper(k_square, input_string))
    for cur_string in [''.join(token) for token in group_list][1:]:
        cur_parent, cur_parent_pos = str_pos_queue.popleft()
        # if value is 0, it cannot be parent node -> skip
        while((cur_parent == '0') and (len(str_pos_queue) > 0)):
            cur_parent, cur_parent_pos = str_pos_queue.popleft()
        # i: order of the child node in the same parent
        cur_pos = [cur_parent_pos*10+i for i in range(1,k_square+1)]
        # pos_list: final position of each node
        pos_list.extend(cur_pos)
        if is_mol:
            str_pos_queue.extend([(s, c) for s, c in zip(grouper_mol(cur_string)[0], cur_pos)])
        else:
            str_pos_queue.extend([(s, c) for s, c in zip(cur_string, cur_pos)])
    
    pos_list = [str(pos) for pos in pos_list]
    # prefix: diagonal
    prefixes = [str((i-1)*k + i) for i in range(1,k+1)]
    # posfix: upper diagonal
    postfixes = []
    for i in range(1,k+1):
        for j in range(i+1, k+1):
            postfixes.append(str((i-1)*k+j))
    
    # find positions ends with 2 including only 1 and 4
    remove_pos_prefix_list = [pos for i, pos in enumerate(pos_list) 
                                if (pos[-1] in postfixes) and len((set(pos[:-1]))-set(prefixes))==0]
    remain_pos_index = [not pos.startswith(tuple(remove_pos_prefix_list)) for pos in pos_list]
    remain_pos_list = [pos for pos in pos_list if not pos.startswith(tuple(remove_pos_prefix_list))]
    
    # find cutting points (one block)
    cut_list = [i for i, pos in  enumerate(remain_pos_list) if pos[-1] == str(k**2)]
    cut_list_2 = [0]
    cut_list_2.extend(cut_list[:-1])
    cut_size_list = [i - j for i, j in zip(cut_list , cut_list_2)]
    cut_size_list[0] += 1
    if is_mol:
        final_string_list = list(compress([item for sublist in grouper_mol(input_string) for item in sublist], remain_pos_index))
    else:
        final_string_list = list(compress([*input_string], remain_pos_index))

    pos_list_iter = iter(final_string_list)
    final_string_cut_list = [list(islice(pos_list_iter, i)) for i in cut_size_list]
    
    return [''.join(l) for l in final_string_cut_list]

def get_max_len(data_name, order='C-M', k=2):
    total_strings = []
    k_square = k**2
    for split in ['train', 'test', 'val']:
        string_path = os.path.join(DATA_DIR, f"{data_name}/{order}/{data_name}_str_{split}_{k}.txt")
        strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        
        total_strings.extend(strings)
    
    red_list = [remove_redundant(string, is_mol=False, k=k) for string in total_strings]
    # red_strings = [''.join(red) for red in red_list]
    
    max_len = max([len(string) for string in total_strings])
    group_max_len = max_len / k_square
    red_max_len = max([len(string) for string in red_list])
    
    return max_len, group_max_len, red_max_len
    