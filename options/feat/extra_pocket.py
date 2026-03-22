# 基于AlphaFill的输出获取口袋
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import struct
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio.PDB import MMCIFParser, Select, MMCIFIO  # type: ignore
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
    model = MMCIFParser(QUIET=True).get_structure("complex", cif_path)[0]  # type: ignore

    ligand_heavy_atoms = []
    model_heavy_atoms = []
    h_atoms = {"H", "D"}
    for res in model.get_residues():
        is_ligand = res.resname.strip() == comp_id and res.id[0].strip() != ""
        for atom in res:
            if atom.element.upper() in h_atoms:
                continue
            if is_ligand:
                ligand_heavy_atoms.append(atom)
            else:
                model_heavy_atoms.append(atom)

    ns = NeighborSearch(model_heavy_atoms)
    pocket_res_keys = set()

    for l_atom in ligand_heavy_atoms:
        close_resi = ns.search(l_atom.coord, radius, level="R")

        for c_res in close_resi:
            c_chain = c_res.get_parent()  # type: ignore
            if c_res.resname != comp_id and is_aa(c_res, standard=False):  # type: ignore
                pocket_res_keys.add(f"{c_chain.id}_{c_res.id[1]}")  # type: ignore

    return pocket_res_keys


def get_poc_resi(cif_path, pocket_residues, save_path):
    if len(pocket_residues) == 0:
        return
    cif_parser = MMCIFParser(QUIET=True)
    io = MMCIFIO()
    structure = cif_parser.get_structure("complex", cif_path)[0]  # type: ignore
    make_dir(os.path.dirname(save_path))
    io.set_structure(structure)
    io.save(save_path, PocketSelect(pocket_residues))


def wapper(args):
    af_path, afill_path, res0compid, save_path = args
    if os.path.exists(save_path):
        return
    if afill_path is not None:
        pocket_residues = extract_heavy_atom_pocket(afill_path, res0compid, 6.5)
        get_poc_resi(af_path, pocket_residues, save_path)
    else:
        get_poc_resi(af_path, res0compid, save_path)

def extra_pocket(sites_file, save_dir):
    data_dict = pkl_load(sites_file)
    tasks = []

    for uid, site_info in data_dict.items():
        residues = site_info["residues"]
        comp_id = site_info["comp_id"]
        
        if residues is not None:
            save_path = f"{save_dir}/{uid2path(uid)}"
            tasks.append((f"{AFDB}/{uid2path(uid)}", None, residues, save_path))

        elif comp_id is not None:
            save_path = f"{save_dir}/{uid2path(uid)}"
            tasks.append((f"{AFDB}/{uid2path(uid)}", f"{AFILLDB}/{uid2path(uid)}", comp_id, save_path))

    pool = mlc.SuperPool(n_cpu=64)
    results = pool.map(wapper, tasks, description="Extracting pocket", chunksize=32)


if __name__ == "__main__":
    # defined_pocket_idx(f"{root_path}/data/enzyme/RHEA/processed/final_sites.pkl", f"{root_path}/data/enzyme/RHEA/processed/uid_rxn_idx.pkl")
    # 273741
    extra_pocket(f"{root_path}/data/enzyme/RHEA/proc/final_sites.pkl",f"{root_path}/data/pocdb")
