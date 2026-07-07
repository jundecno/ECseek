import rootutils
root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))

from utils import *
from torch_geometric.nn import radius_graph
import torch.nn.functional as F
from torch_geometric.data import Data

def get_poc_coors(poc_struct):
    poc_coords = []
    for res in poc_struct.get_residues():  # type: ignore
        atoms = {atom.get_name(): atom for atom in res.get_atoms()}  # type: ignore
        all_coords = [atom.get_coord() for atom in atoms.values()]
        mean_coord = np.mean(all_coords, axis=0) if all_coords else np.zeros(3)
        N = atoms["N"].get_coord() if "N" in atoms else mean_coord
        CA = atoms["CA"].get_coord() if "CA" in atoms else mean_coord
        C = atoms["C"].get_coord() if "C" in atoms else mean_coord
        O = atoms["O"].get_coord() if "O" in atoms else mean_coord
        # side chain 质心
        side_atoms = [atom for name, atom in atoms.items() if name not in main_enum]  # type: ignore
        # 计算侧链原子坐标的质心 elemmass获取质量
        if len(side_atoms) > 0:
            coords = np.array([atom.get_coord() for atom in side_atoms])
            masses = np.array([elem2mass(atom.element) for atom in side_atoms]).reshape(-1, 1)
            total_mass = np.sum(masses)
            side_mean_center = np.sum(coords * masses, axis=0) / total_mass if total_mass > 0 else CA
        else:
            side_mean_center = CA
        poc_coords.append((N, CA, C, O, side_mean_center))
    return np.array(poc_coords, dtype=np.float32)


def get_pre_feat(poc_file, prot_pre_dir):
    uid = os.path.basename(poc_file)
    poc_struct = load_structure(poc_file)
    # sequence
    poc_indices = [res.id[1] - 1 for res in poc_struct.get_residues()] # type: ignore
    prot_res_feat = np.load(f"{prot_pre_dir}/{uid2path(uid,True)}.npz")
    poc_pre_feat = prot_res_feat["node_feature"][poc_indices].astype(np.float32)
    # structure
    coords = get_poc_coors(poc_struct)
    # save
    return poc_pre_feat, coords


def get_protein_geo_edge(pos, cutoff=8.0, self_loops=False):
    X_ca = pos[:, 1, :]  # [num_res, 3]
    edge_index = radius_graph(X_ca, r=cutoff, loop=self_loops, max_num_neighbors=500, num_workers=4)
    geo_prot_feat, edge_attr = get_geo_feat(pos, edge_index)
    return geo_prot_feat, edge_index, edge_attr


def get_geo_feat(X, edge_index):
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    if D.size(-1) != 1:
        D = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D - D_mu) / D_sigma) ** 2))
    return RBF


def _get_direction_orientation(X, edge_index):  # N, CA, C, O, R
    X_N = X[:, 0]  # [L, 3]
    X_Ca = X[:, 1]
    X_C = X[:, 2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v, dim=-1), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=-1)  # [L, 3, 3] (3 column vectors)
    node_j, node_i = edge_index
    t = F.normalize(X[:, [0, 2, 3, 4]] - X_Ca.unsqueeze(1), dim=-1)  # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1)  # [L, 4 * 3]
    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1)  # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1)  # [E, 5 * 3] # slightly improve performance
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim=-1)  # [E, 2 * 5 * 3]
    r = torch.matmul(local_frame[node_i].transpose(-1, -2), local_frame[node_j])  # [E, 3, 3]
    edge_orientation = _quaternions(r)  # [E, 4]
    return node_direction, edge_direction, edge_orientation


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)))
    _R = lambda i, j: R[:, i, j]
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q


def _get_distance(X, edge_index):
    idx1 = [1, 1, 1, 0, 0, 3, 4, 4, 4, 4]
    idx2 = [0, 2, 3, 2, 3, 2, 0, 1, 2, 3]
    atoms_a = X[:, idx1]
    atoms_b = X[:, idx2]
    dists_node = (atoms_a - atoms_b).norm(dim=-1)
    node_dist = _rbf(dists_node).flatten(1)
    X_src = X[edge_index[0]]
    X_dst = X[edge_index[1]]
    diff = X_src.unsqueeze(2) - X_dst.unsqueeze(1)
    dists_edge = diff.norm(dim=-1)  # [E, 5, 5]
    edge_dist = _rbf(dists_edge).flatten(1)
    return node_dist, edge_dist


def poc_graph(uid_file, poc_dir, mean_file, save_dir):
    uid_list = read_lines(uid_file)
    esm_dict = pkl_load(file_path=mean_file)
    for uid in tqdm(uid_list):
        file = f"{poc_dir}/{uid2path(uid,True)}.cif"
        if os.path.exists(f"{save_dir}/{uid2path(uid,True)}.pkl"):
            continue
        if os.path.exists(file):
            # seq coords
            seq_feat, coords = get_pre_feat(file, f"{root_path}/data/features/protein")
        else:
            seq_feat = esm_dict[uid].astype(np.float32).reshape(1, -1)
            coords = np.zeros((1, 5, 3), dtype=np.float32)

        # 根据位置计算图
        seqs = torch.tensor(seq_feat, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        geo_node_feat, edge_index, geo_edge_feat = get_protein_geo_edge(coords, cutoff=8.0, self_loops=True)
        node = torch.cat([seqs, geo_node_feat], dim=-1)  # [n, 1324]
        global_feat = torch.tensor(esm_dict[uid], dtype=torch.float32).reshape(1, -1)  # [1, 1152]
        poc_graph_data = Data(x=node, edge_index=edge_index, edge_attr=geo_edge_feat, seq=global_feat)
        make_dir(os.path.dirname(f"{save_dir}/{uid2path(uid,True)}.pkl"))
        pkl_dump(f"{save_dir}/{uid2path(uid,True)}.pkl", poc_graph_data)


def poc_graph2lmdb(pocket_dir, save_dir):
    files = get_file_paths(pocket_dir)
    env = lmdb.open(save_dir, map_size=1024**4, map_async=True, writemap=True)
    txn = env.begin(write=True)
    commit_every = 1000
    all_uids = []

    for i, file in enumerate(tqdm(files, desc="Processing Graphs")):
        uid = os.path.basename(file)
        data = pkl_load(file)

        txn.put(uid.encode("utf-8"), pkl.dumps(data, protocol=pkl.HIGHEST_PROTOCOL))
        all_uids.append(uid)

        if (i + 1) % commit_every == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b"__keys__", pkl.dumps(all_uids, protocol=pkl.HIGHEST_PROTOCOL))
    txn.put(b"__len__", str(len(all_uids)).encode("utf-8"))
    txn.commit()
    env.close()
    print(f"LMDB 写入完成，共计 {len(all_uids)} 条数据。")


if __name__ == "__main__":
    poc_graph(
        f"{root_path}/data/enzyme/RHEA/split/all_uids.txt",
        f"{root_path}/data/features/pocdb/",
        f"{root_path}/data/features/esm_mean_feat.pkl",
        f"{root_path}/data/features/pocket_graph/",
    )
    # poc_graph2lmdb(f"{root_path}/data/features/pocket_graph/", f"{root_path}/data/features/pocket_graph.lmdb")