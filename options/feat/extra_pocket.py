# 基于AlphaFill的输出获取口袋
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import struct
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio.PDB import MMCIFParser, Select, MMCIFIO # type: ignore
from Bio.PDB.Polypeptide import is_aa

# 利用site提取口袋
def defined_pocket_idx(sites_file, save_file):
    data_dict = pkl_load(sites_file)
    keys = list(data_dict.keys())
    # sort
    keys = sorted(keys, key=lambda x: (x[0], x[1]))
    uid_rxn_idx = {}
    for key in tqdm(keys):
        uid, rxn_smiles = key
        if uid not in uid_rxn_idx:
            uid_rxn_idx[uid] = {}
        uid_rxn_idx[uid][rxn_smiles] = len(uid_rxn_idx[uid])
    pkl_dump(save_file, uid_rxn_idx)


class PocketSelect(Select):
    def __init__(self, pocket_residues):
        self.pocket_residues = set(pocket_residues)

    def accept_residue(self, residue):  # type: ignore
        chain_resi = f"{residue.get_parent().id}_{residue.id[1]}"
        return chain_resi in self.pocket_residues


from Bio.PDB.NeighborSearch import NeighborSearch

def extract_heavy_atom_pocket(cif_path, comp_id, radius=8.0):
    model = MMCIFParser(QUIET=True).get_structure("complex", cif_path)[0] # type: ignore

    ligand_heavy_atoms = []
    model_heavy_atoms = []

    for chain in model:
        for res in chain:
            is_ligand = res.resname == comp_id
            for atom in res:
                if atom.element.upper() not in ["H", "D"]:
                    model_heavy_atoms.append(atom)
                    if is_ligand:
                        ligand_heavy_atoms.append(atom)

    if not ligand_heavy_atoms:
        return None

    ns = NeighborSearch(model_heavy_atoms)
    pocket_res_keys = set()

    for l_atom in ligand_heavy_atoms:
        close_resi = ns.search(l_atom.coord, radius, level="R")

        for c_res in close_resi:
            c_chain = c_res.get_parent()
            if c_res.resname != comp_id and is_aa(c_res, standard=False):
                pocket_res_keys.add(f"{c_chain.id}_{c_res.id[1]}")

    if not pocket_res_keys:
        return None

    return pocket_res_keys

def resi_thread(cif_path, pocket_residues, save_path):
    cif_parser = MMCIFParser(QUIET=True)
    io = MMCIFIO()
    structure = cif_parser.get_structure("complex", cif_path)[0] # type: ignore
    make_dir(os.path.dirname(save_path))
    io.set_structure(structure)
    io.save(save_path, PocketSelect(pocket_residues))

def comp_thread(af_path, cif_path, comp_id, save_path):
    pocket_residues = extract_heavy_atom_pocket(cif_path, comp_id, 6.5)
    if pocket_residues is not None:
        resi_thread(af_path, pocket_residues, save_path)

def extra_pocket(sites_file, uid_rxn_idx, save_dir):
    data_dict = pkl_load(sites_file)
    uid_rxn_idx_dict = pkl_load(uid_rxn_idx)
    res_arr = []
    pool = Pool(processes=16)
    for key, site_info in data_dict.items():
        uid, rxn_smiles = key
        residues = site_info["residues"]
        comp_id = site_info["comp_id"]
        if residues is not None:
            save_path = f"{save_dir}/{uid2path(uid,True)}/pocket_{uid_rxn_idx_dict[uid][rxn_smiles]}.cif"
            # executor.submit(resi_thread, f"{AFDB}/{uid2path(uid)}", residues, save_path)
            res = pool.apply_async(resi_thread, args=(f"{AFDB}/{uid2path(uid)}", residues, save_path))
            res_arr.append(res)
        if comp_id is not None:
            save_path = f"{save_dir}/{uid2path(uid,True)}/pocket_{uid_rxn_idx_dict[uid][rxn_smiles]}.cif"
            # executor.submit(comp_thread, f"{AFDB}/{uid2path(uid)}", f"{AFILLDB}/{uid2path(uid)}", comp_id, save_path)
            res = pool.apply_async(comp_thread, args=(f"{AFDB}/{uid2path(uid)}", f"{AFILLDB}/{uid2path(uid)}", comp_id, save_path))
            res_arr.append(res)
    pool.close()
    success_count = 0
    error_logs = []

    for res_obj in tqdm(res_arr, desc="Processing results"):
        try:
            is_success, msg = res_obj.get()
            if is_success:
                success_count += 1
            else:
                error_logs.append(msg)
        except Exception as e:
            error_logs.append(f"子进程发生致命崩溃: {str(e)}")

    # 等待所有子进程彻底退出
    pool.join()


if __name__ == "__main__":
    # defined_pocket_idx(f"{root_path}/data/enzyme/RHEA/processed/final_sites.pkl", f"{root_path}/data/enzyme/RHEA/processed/uid_rxn_idx.pkl")
    # 273741
    extra_pocket(
        f"{root_path}/data/enzyme/RHEA/processed/final_sites.pkl",
        f"{root_path}/data/enzyme/RHEA/processed/uid_rxn_idx.pkl",
        f"{root_path}/data/pockets",
    )
