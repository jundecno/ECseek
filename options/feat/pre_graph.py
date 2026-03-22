import itertools

import drfp
import rootutils
import torch

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from torch_geometric.nn import radius_graph, radius
import torch.nn.functional as F
from torch_geometric.data import Data
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


def poc_graph(poc_feat_dir, save_dir):
    files = get_file_paths(poc_feat_dir)
    for file in tqdm(files):
        uid = os.path.basename(file).split(".")[0]
        save_file = os.path.join(save_dir, uid2path(uid,True)+".pkl")
        make_dir(os.path.dirname(save_file))
        data = np.load(file)
        # seq coords
        seq_feat = data["seq"] # [n, 1152]
        coords = data["coords"] # [n, 5, 3]
        # 根据位置计算图
        seqs = torch.tensor(seq_feat, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        geo_node_feat, edge_index, geo_edge_feat = get_protein_geo_edge(coords,cutoff=8.0, self_loops=True)
        node = torch.cat([seqs, geo_node_feat], dim=-1) # [n, 1324]
        poc_graph_data = Data(x=node, edge_index=edge_index, edge_attr=geo_edge_feat)
        pkl_dump(save_file, poc_graph_data)


def smi_graph(smis):

    mol = Chem.MolFromSmiles(smis)
    atoms_elem = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        atm1 = bond.GetBeginAtomIdx()
        atm2 = bond.GetEndAtomIdx()
        edge_index[0].append(atm1)  # src j
        edge_index[1].append(atm2)  # dst i

        edge_feature_vector = []
        edge_feature_vector.extend(one_of_k_encoding("covalent", ["self-loop", "covalent", "non-covalent"]))
        edge_feature_vector.extend(one_of_k_encoding(bond.GetBondTypeAsDouble(), [0.0, 1.0, 1.5, 2.0, 3.0]))
        edge_feature_vector.append(bond.GetIsConjugated())
        edge_feature_vector.append(bond.IsInRing())
        edge_feature_vector.extend(one_of_k_encoding(bond.GetStereo(), bonds_allowed))
        # 添加非反应边信息
        edge_feature_vector.extend(one_of_k_encoding("non-reaction", ["non-reaction", "reaction"])) #0, 1
        edge_attr.append(edge_feature_vector)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=atoms_elem, edge_index=edge_index, edge_attr=edge_attr)


def rxn_smi_graph(rxn, rc, undirected=True, self_loops=False):
    subs_smi, prod_smi = rxn.split(">>")
    subs_graph = smi_graph(subs_smi)
    prod_graph = smi_graph(prod_smi)
    rxn_x = torch.cat([subs_graph.x, prod_graph.x], dim=0) # type: ignore
    rxn_edge_index = torch.cat([subs_graph.edge_index, prod_graph.edge_index + subs_graph.x.shape[0]], dim=-1) # type: ignore
    rxn_edge_attr = torch.cat([subs_graph.edge_attr, prod_graph.edge_attr], dim=0)  # type: ignore
    if len(rc[0]) > 0 and len(rc[1]) >0:
        subs = torch.tensor(rc[0])
        prod = torch.tensor(rc[1]) + subs_graph.x.shape[0] # type: ignore
        rc_edge_index = torch.stack([subs.repeat_interleave(len(prod)),prod.repeat(len(subs))], dim=0)
        rc_edge_attr = torch.zeros((rc_edge_index.shape[1], rxn_edge_attr.shape[1]), dtype=torch.float)  # type: ignore
        rc_edge_attr[:, -1] = 1.0  # type: ignore
        edge_index = torch.cat([rxn_edge_index, rc_edge_index], dim=-1)  # type: ignore 2 x
        edge_attr = torch.cat([rxn_edge_attr, rc_edge_attr], dim=0)  # type: ignore x d
        # shape
        edge_index, edge_attr = make_undirected_with_self_loops(edge_index, edge_attr, undirected=undirected, self_loops=self_loops)
    else:
        edge_index, edge_attr = make_undirected_with_self_loops(rxn_edge_index, rxn_edge_attr, undirected=undirected, self_loops=self_loops)
    return rxn_x, edge_index, edge_attr


def rxn_graph(feat_dir, save_dir):
    drfp_dict = pkl_load(os.path.join(feat_dir, "drfp.pkl"))
    rxn_rc_dict = pkl_load(os.path.join(feat_dir, "rxn2rc.pkl"))
    smi_feat_dict = pkl_load(os.path.join(feat_dir, "smi_feat.pkl"))
    rxn_graph_dict = {}
    for rxn in tqdm(rxn_rc_dict):
        # pre feature
        subs, prod = rxn.split(">>")
        subs_smis, prod_smis = subs.split("."), prod.split(".")
        subs_feat = np.concatenate([smi_feat_dict[smi] for smi in subs_smis], axis=0) # [s, d]
        prod_feat = np.concatenate([smi_feat_dict[smi] for smi in prod_smis], axis=0) # [p, d]
        rxn_seq_feat = np.concatenate([subs_feat, prod_feat], axis=0) # [s+p ,d]
        # rxn smiles graph
        rxn_x, edge_index, edge_attr = rxn_smi_graph(rxn, rxn_rc_dict[rxn], undirected=True, self_loops=False)
        drfp_feat = torch.tensor(drfp_dict[rxn], dtype=torch.float32).reshape(1, -1) # [1, d]
        rxn_seq_feat = torch.tensor(rxn_seq_feat, dtype=torch.float32) # [s+p, d]
        rxn_graph_dict[rxn] = Data(x=rxn_x, edge_index=edge_index, edge_attr=edge_attr, drfp=drfp_feat, seq=rxn_seq_feat)
        return
    pkl_dump(os.path.join(save_dir, f"rxn_graph.pkl"), rxn_graph_dict)
if __name__ == "__main__":
    # poc_graph(f"{root_path}/data/features/pocket/", f"{root_path}/data/features/pocket_graph/")
    rxn_graph(f"{root_path}/data/features/", f"{root_path}/data/features/")
    # data = pkl_load("/data/zzjun/ECseek/data/features/rxn_graph.pkl")
    # print(list(data.values())[:10])
