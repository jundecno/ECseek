import lmdb
import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)

from utils import json_load, pkl
from torch_geometric.data import Dataset, Data
import os
from torch_geometric.loader import DataLoader
import torch
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")

def add_virtual_node(data: Data):
    num_nodes = data.num_nodes
    vn_x = torch.mean(data.x, dim=0, keepdim=True)  # type: ignore
    data.x = torch.cat([data.x, vn_x], dim=0)  # type: ignore

    row, col = data.edge_index  # type: ignore
    arange = torch.arange(num_nodes, device=row.device)  # type: ignore
    full = row.new_full((num_nodes,), num_nodes)  # type: ignore
    row = torch.cat([row, arange, full], dim=0)
    col = torch.cat([col, full, arange], dim=0)
    data.edge_index = torch.stack([row, col], dim=0)

    num_new_edges = 2 * num_nodes  # type: ignore
    vn_edge_attr = data.edge_attr.new_zeros((num_new_edges, data.edge_attr.size(1)))  # type: ignore
    data.edge_attr = torch.cat([data.edge_attr, vn_edge_attr], dim=0)  # type: ignore

    data.num_nodes = num_nodes + 1  # type: ignore
    return data


class TrainDataset(Dataset):

    def __init__(self, feat_dir, pairs):
        super().__init__()
        self.rxn2idx_dict = json_load(os.path.join(feat_dir, "rxn2idx.json"))
        self.feat_dir = feat_dir
        self.pairs = pairs
        self.poc_env = None
        self.rxn_env = None
        self.poc_txn = None
        self.rxn_txn = None

    def len(self):
        return len(self.pairs)

    def _init_db(self):
        if self.poc_env is None:
            self.poc_env = lmdb.open(
                os.path.join(self.feat_dir, "poc_graph.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        if self.rxn_env is None:
            self.rxn_env = lmdb.open(
                os.path.join(self.feat_dir, "rxn_graph.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )

    def get(self, idx):  # type: ignore
        self._init_db()

        uid, rxn_id = self.pairs[idx]
        rxn_idx = self.rxn2idx_dict[rxn_id]

        with self.poc_env.begin(write=False) as txn:  # type: ignore
            poc_file = txn.get(uid.encode("utf-8"))
            poc_graph = pkl.loads(poc_file)  # type: ignore
            poc_graph.id = uid

        with self.rxn_env.begin(write=False) as txn:  # type: ignore
            rxn_file = txn.get(str(rxn_idx).encode("utf-8"))
            rxn_graph = pkl.loads(rxn_file)  # type: ignore
            rxn_graph.id = rxn_id

        return poc_graph, rxn_graph


class TestDataset(Dataset):
    def __init__(self, feat_dir, info_list, info_type="rxn"):
        super().__init__()
        self.rxn2idx_dict = json_load(os.path.join(feat_dir, "rxn2idx.json"))
        self.feat_dir = feat_dir
        self.info_list = info_list
        self.info_type = info_type

        self.poc_env = None
        self.rxn_env = None

    def len(self):
        return len(self.info_list)

    def _init_db(self):
        if self.poc_env is None and self.info_type == "poc":
            self.poc_env = lmdb.open(
                os.path.join(self.feat_dir, "poc_graph.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        if self.rxn_env is None and self.info_type == "rxn":
            self.rxn_env = lmdb.open(
                os.path.join(self.feat_dir, "rxn_graph.lmdb"),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )

    def get(self, idx):  # type: ignore
        self._init_db()
        info = self.info_list[idx]

        if self.info_type == "poc":
            with self.poc_env.begin(write=False) as txn:  # type: ignore
                poc_file = txn.get(info.encode("utf-8"))
                graph = pkl.loads(poc_file)  # type: ignore
                graph.type = "poc"
            return graph

        elif self.info_type == "rxn":
            rxn_idx = self.rxn2idx_dict[info]
            with self.rxn_env.begin(write=False) as txn:  # type: ignore
                rxn_file = txn.get(str(rxn_idx).encode("utf-8"))
                graph = pkl.loads(rxn_file)  # type: ignore
                graph.type = "rxn"
            return graph


class PocDataset(Dataset):
    def __init__(self, lmdb_file, info_type="poc"):
        super().__init__()
        self.lmdb_file = lmdb_file
        self.info_type = info_type
        self.poc_env = lmdb.open(self.lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        with self.poc_env.begin(write=False) as txn:  # type: ignore
            self.info_list = [key.decode("utf-8") for key, _ in txn.cursor()]  # type: ignore

    def len(self):
        return len(self.info_list)

    def get(self, idx):  # type: ignore
        info = self.info_list[idx]

        if self.info_type == "poc":
            with self.poc_env.begin(write=False) as txn:  # type: ignore
                poc_file = txn.get(info.encode("utf-8"))
                graph = pkl.loads(poc_file)  # type: ignore
                graph.type = "poc"
            return graph
