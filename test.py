
import torch
import numpy as np
from data.load_data import generate_string
import os
import networkx as nx
from data.data_utils import get_max_len


for data in ['point']:
    print(get_max_len(data, 'C-M', 2))
