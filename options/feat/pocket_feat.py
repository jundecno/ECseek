import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio.PDB import MMCIFParser # type: ignore
# 按照残基索引获取poc esm feature
def get_poc_idx_af2(poc_struct):
    poc_res_ids = [res.id[1] for res in poc_struct.get_residues()] # type: ignore
    poc_indices = [res_id - 1 for res_id in poc_res_ids]
    return poc_res_ids


def get_poc_coors(poc_struct):
    poc_coords = []
    for res in poc_struct.get_residues(): # type: ignore
        atoms = {atom.get_name(): atom for atom in res.get_atoms()} # type: ignore
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


def create_pocket_feat(poc_dir, prot_pre_dir, save_dir):
    poc_files = get_file_paths(poc_dir)
    parser = MMCIFParser(QUIET=True)
    for poc_file in tqdm(poc_files):
        uid = os.path.basename(poc_file)
        save_path = f"{save_dir}/{uid2path(uid,True)}.npz"
        if os.path.exists(save_path):
            continue
        try:
            poc_struct = parser.get_structure("poc", poc_file)
            # sequence
            poc_indices = get_poc_idx_af2(poc_struct)
            prot_res_feat = np.load(f"{prot_pre_dir}/{uid2path(uid,True)}.npz")
            poc_pre_feat = prot_res_feat["node_feature"][poc_indices].astype(np.float32)
            # structure
            coords = get_poc_coors(poc_struct)
            # save
            make_dir(os.path.dirname(save_path))
            np.savez(save_path, seq=poc_pre_feat, coords=coords)
        except Exception as e:
            print(f"Failed to process {poc_file} {e}")


def fill_nan_poc(esm_file, save_dir):
    esm_mean_dict = pkl_load(esm_file)
    for uid in tqdm(esm_mean_dict):
        save_file = f"{save_dir}/{uid2path(uid,True)}.npz"
        if os.path.exists(save_file):
            continue
        else:
            poc_pre_feat = esm_mean_dict[uid].astype(np.float32).reshape(1, -1)
            coords = np.zeros((1, 5, 3), dtype=np.float32)
            make_dir(os.path.dirname(save_file))
            np.savez(save_file, seq=poc_pre_feat, coords=coords)


if __name__ == "__main__":
    # 读取poc esm feature
    # print(len(list(get_files("/data/zzjun/ECseek/data/pockets/"))))
    # print(len(list(get_files("/data/zzjun/ECseek/data/features/protein"))))

    # print(get_poc_idx_af2(parser.get_structure("poc", "/data/zzjun/ECseek/data/pocdb/A0/A0/09/A0A009IHW8.cif")))
    create_pocket_feat(f"{root_path}/data/pocdb/", f"{root_path}/data/features/protein", f"{root_path}/data/features/pocket")
    # data = np.load("/data/zzjun/ECseek/data/features/pocket/A0/A0/09/A0A009IHW8.npz")
    # print(data["seq"].shape)
    fill_nan_poc(f"{root_path}/data/features/esm_mean_feat.pkl", f"{root_path}/data/features/pocket")
