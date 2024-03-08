import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from tqdm import tqdm

from data.data_utils import remove_redundant
from data.tokens import tokenize


DATA_DIR = "resource"
    
class GenericDataset(Dataset):
    is_mol = False
    def __init__(self, split, string_type='group-red', order='C-M', is_tree=False, k=2):
        self.string_type = string_type
        self.is_tree = is_tree
        self.order = order
        self.k = k
        if k > 2:
            string_path = os.path.join(self.raw_dir, f"{self.order}/{self.data_name}_str_{split}_{self.k}.txt")
        else:
            string_path = os.path.join(self.raw_dir, f"{self.order}/{self.data_name}_str_{split}.txt")
        self.strings = Path(string_path).read_text(encoding="utf=8").splitlines()

        # remove redundant
        self.strings = [remove_redundant(string, self.is_mol, self.k) for string in tqdm(self.strings, 'Removing redundancy')]
        
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type, self.k))
    
class ComDataset(GenericDataset):
    data_name = 'GDSS_com'
    raw_dir = f'{DATA_DIR}/GDSS_com'
    
class EnzDataset(GenericDataset):
    data_name = 'GDSS_enz'
    raw_dir = f'{DATA_DIR}/GDSS_enz'

class GridDataset(GenericDataset):
    data_name = 'GDSS_grid'
    raw_dir = f'{DATA_DIR}/GDSS_grid'


class QM9Dataset(GenericDataset):
    data_name = "qm9"
    raw_dir = f"{DATA_DIR}/qm9"
    is_mol = True
        
class ZINCDataset(GenericDataset):
    data_name = 'zinc'
    raw_dir = f'{DATA_DIR}/zinc'
    is_mol = True
    
class PlanarDataset(GenericDataset):
    data_name = 'planar'
    raw_dir = f'{DATA_DIR}/planar'
            