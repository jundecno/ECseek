from math import e

import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *

# 读取lmdb文件，修改
def fix_poc_graph(file):
    env = lmdb.open(str(file), subdir=False, lock=False, readahead=False, meminit=False, max_readers=64, map_size=1099511627776)
    with env.begin(write=True) as txn:
        n_entries = txn.stat()["entries"]
        cursor = txn.cursor()
        for key, value in tqdm(cursor, total=n_entries, desc="Fix pocket graph"):
            graph = pkl.loads(value)
            # 修改空图为单节点图
            if graph.x.size(0) == 0:
                graph.x = torch.zeros((1, 960), dtype=torch.float)
                graph.dssp_x = torch.zeros((1, 16), dtype=torch.float)
                graph.sym_x = torch.zeros((1,), dtype=torch.long)
                graph.phy_x = torch.zeros((1, 6), dtype=torch.float)
                graph.ca_x = torch.zeros((1, 3), dtype=torch.float)
                graph.edge_index = torch.zeros((2, 0), dtype=torch.long)
                graph.edge_attr = torch.zeros((0, 22), dtype=torch.float)
                txn.put(key, pkl.dumps(graph))
    env.sync()
    env.close()

if __name__ == "__main__":
    fix_poc_graph(f"{root_path}/data/features/poc_graph.lmdb")
