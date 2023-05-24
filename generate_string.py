import argparse

from data.load_data import generate_string
from data.load_data import generate_mol_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset_name: GDSS_com, GDSS_enz, GDSS_grid, planar, qm9, zinc
    parser.add_argument("dataset_name", type=str, default='GDSS_com')
    # order: C-M, BFS, DFS
    parser.add_argument("order", type=str, default='C-M')
    # k: 2, 3
    parser.add_argument("k", type=int, default=2)
    
    args, _ = parser.parse_known_args()
    
    if args.dataset_name in ['GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar']:
        generate_string(dataset_name=args.dataset_name, order=args.order, k=args.k)
    else:
        generate_mol_string(dataset_name=args.dataset_name, order=args.order)