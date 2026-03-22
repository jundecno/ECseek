import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)

from utils import pkl_load, uid2path
import torch
from torch_geometric.data import Dataset, HeteroData, Data
from torch_geometric.loader import DataLoader
import os

class RHEADataset(Dataset):
    def __init__(self, feat_dir, pairs):
        super().__init__()
        self.poc_dir = os.path.join(feat_dir, "pocket_graph")
        self.rxn_graph = pkl_load(os.path.join(feat_dir, "rxn_graph.pkl"))
        self.pairs = pairs

    def len(self):
        return len(self.pairs)

    def get(self, idx):  # type: ignore
        uid, pos_rxn_id, neg_rxn_id = self.pairs[idx]
        poc_graph = pkl_load(os.path.join(self.poc_dir, f"{uid2path(uid, True)}.pkl"))
        pos_rxn_graph = self.rxn_graph[pos_rxn_id]
        neg_rxn_graph = self.rxn_graph[neg_rxn_id]
        return poc_graph, pos_rxn_graph, neg_rxn_graph


