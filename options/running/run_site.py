# 使用pocketeer预测后，使用p2rank进行结构评估，选取top1的结构进行后续分析
from concurrent.futures import ThreadPoolExecutor
from multiprocess import Pool
import rootutils


root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import ast
from rdkit.Chem import rdMolDescriptors
from Bio.PDB.Polypeptide import is_aa

def get_ligands(cif_file):
    model = MMCIFParser(QUIET=True).get_structure("complex", cif_file)[0]  # type: ignore
    ligands = set()
    uid = os.path.basename(cif_file).split(".")[0]
    for res in model.get_residues():
        hetflag = res.id[0].strip()
        if hetflag == "" or is_aa(res, standard=True):
            continue
        ligands.add(res.resname.strip())

    return uid, list(ligands)


def get_afill_ligands(afill_dir, save_file):
    files = []
    all_files = tranverse_folder(afill_dir)
    for filename in all_files:
        if ".cif" in filename:
            files.append(filename)
    ligand_dict = {}
    pool = mlc.SuperPool(64)
    results = pool.map(get_ligands, files)
    for uid, ligands in results:
        ligand_dict[uid] = ligands
    # save csv file
    pd.DataFrame({"uid": list(ligand_dict.keys()), "ligands": list(ligand_dict.values())}).to_csv(save_file, index=False)


def get_similar_based_on_rxn_and_afill(rxn2uid, rxn2smi, afill_ligands_csv, comp_id2smi, save_file):
    uid2rxn = pkl_load(rxn2uid)
    rxn_smis = json_load(rxn2smi)  # for smiles from rxn
    afill_ligands = pd.read_csv(afill_ligands_csv, converters={"ligands": ast.literal_eval})
    afill_ligands = dict(zip(afill_ligands["uid"].values, afill_ligands["ligands"].values))  # for rcsb
    comp_id2smi_dict = json_load(comp_id2smi)  # for smiles from comp_id
    res_dict = pkl_load(save_file) if os.path.exists(save_file) else {}

    for uid, rxn_smi_ids in tqdm(uid2rxn.items()):
        if uid in res_dict:
            continue
        smis_from_rxn = []
        for rxn_smi_id in rxn_smi_ids:
            smis_from_rxn.extend(rxn_smis[rxn_smi_id])
        atom_max = -1
        rxn_smiles_best = ""
        for rxn_smiles_str in smis_from_rxn:
            mol = Chem.MolFromSmiles(rxn_smiles_str, sanitize=False)
            if mol is None:
                continue
            atom_num = mol.GetNumHeavyAtoms()
            if atom_num > atom_max:
                atom_max = atom_num
                rxn_smiles_best = rxn_smiles_str

        if atom_max == -1:
            res_dict[uid] = {"smiles_rxn": "", "smiles_afill": [], "sims": np.array([])}
            continue
        # 计算rxn_smiles_best与afill_ligands中所有ligand smiles的相似度，找到最相似的ligand smiles
        smis_from_afill = [comp_id2smi_dict[comp_id] for comp_id in afill_ligands.get(uid, [])]
        smis_sims = calculate_top_matrix(rxn_smiles_best, smis_from_afill)
        res_dict[uid] = {"smiles_rxn": smis_sims[0], "smiles_afill": smis_sims[1], "sims": smis_sims[2]}
    pkl_dump(save_file, res_dict)


def run_pocketeer(args):
    cif_file, tmp_dir = args
    base_name = os.path.basename(cif_file).split(".")[0]
    if os.path.exists(f"{tmp_dir}/{base_name}_pocketeer/pockets.json"):
        return
    try:
        os.system(f"pocketeer {cif_file} -o {tmp_dir}/{base_name}_pocketeer > /dev/null 2>&1")
    except:
        return


def batch_run_site(rxn_afill_ligands):
    rxn_afill_ligands_dict = pkl_load(rxn_afill_ligands)
    tmp_dir = f"{root_path}/data/tmp/pocketeer"
    res_afill_dict = pkl_load(f"{root_path}/data/enzyme/RHEA/proc/sites_afill.pkl") if os.path.exists(f"{root_path}/data/enzyme/RHEA/proc/sites_afill.pkl") else {}
    cif_files = []
    for uid, data in tqdm(rxn_afill_ligands_dict.items()):
        if uid in res_afill_dict:
            continue
        rxn_smi = data["smiles_rxn"]
        afill_smi = data["smiles_afill"]
        sims = data["sims"]
        # 逻辑：如果原子数目大于10，选择最相似的AlphaFill填充的配体。
        if sims.size != 0:
            afill_smi_best = afill_smi[sims.argmax()]
            res_afill_dict[uid] = {"smiles_rxn": rxn_smi, "smiles_afill": afill_smi_best}
        else:  # 使用prank进行预测
            cif_file = f"{AFDB}/{uid2path(uid)}"
            cif_files.append(cif_file)
    # run pocketeer for those cif files
    pool = mlc.SuperPool(64)
    results = pool.map(run_pocketeer, [(cif_file, tmp_dir) for cif_file in cif_files])

    line_str = ""
    for file in os.listdir(tmp_dir):
        uid = file.split("_")[0]
        save_file = f"{tmp_dir}/{file}/pockets.json"
        cif_file = f"{AFDB}/{uid2path(uid)}".replace("_pocketeer", "")
        if os.path.exists(f"{tmp_dir.replace("pocketeer", "prank_teer")}/{uid}.cif_predictions.csv"):
            continue
        if os.path.exists(save_file):
            line_str += f"{save_file} {cif_file}\n"

    pkl_dump(f"{root_path}/data/enzyme/RHEA/proc/sites_afill.pkl", res_afill_dict)
    # write ds
    ds_str = "PARAM.PREDICTION_METHOD=pocketeer\n\nHEADER: prediction protein\n\n" + line_str
    write_txt(f"{root_path}/data/enzyme/RHEA/proc/pocketeer_prank.ds", ds_str)
    os.system(
        f"{PRANK_BIN} rescore {root_path}/data/enzyme/RHEA/proc/pocketeer_prank.ds -c rescore_2024 -o {tmp_dir.replace("pocketeer", "prank_teer")}"
    )


def batch_run_prank(prank_path, save_path):
    # 如果pocketeer没有预测出结合位点，则使用prank进行预测
    cmd_strs = []
    for file in os.listdir(prank_path):
        if "_predictions.csv" in file:
            df = pd.read_csv(f"{prank_path}/{file}", sep=r"\s*,\s*", engine="python")
            uid = file.split(".")[0]
            # 如果pocketeer没有预测出结合位点，则说明该蛋白质没有已知的配体结合位点, 可能是因为该蛋白质没有已知的配体结合位点, 或者pocketeer没有找到任何匹配的模板结构.
            if df.shape[0] == 0:
                cmd_strs.append(f"{PRANK_BIN} predict -f {AFDB}/{uid2path(uid)} -o {save_path}/{uid} > /dev/null 2>&1")
    pool = mlc.SuperPool(64)
    results = pool.map(os.system, cmd_strs)

def merge_all_sites(rxn2uid, afill_sites, pocketeer_dir, prank_dir, afill_ligands, save_file):
    # 确定逻辑：优先选择afill，否则选择pocketeer，如果pocketeer没有预测出结合位点，则选择prank，最后使用afill填充的配体
    # 确定最终的结合位点文件格式，包含uid, rxn_smiles, residues, compid
    rxn2uid_df = pkl_load(rxn2uid)
    uids = rxn2uid_df.keys()
    afill_sites_dict = pkl_load(afill_sites)
    afill_ligands_dict = pd.read_csv(afill_ligands, converters={"ligands": ast.literal_eval})
    afill_ligands_dict = dict(zip(afill_ligands_dict["uid"].values, afill_ligands_dict["ligands"].values))
    compid_dict = json_load(f"{root_path}/data/enzyme/RHEA/proc/comp_id2smi.json")
    compid_reverse_dict = {v: k for k, v in compid_dict.items()}

    res_dict = {}
    for uid in tqdm(uids):
        if uid in afill_sites_dict:
            ligand_smiles = afill_sites_dict[uid]["smiles_afill"]
            comp_id = compid_reverse_dict[ligand_smiles]
            res_dict[uid] = {"residues": None, "comp_id": comp_id}
        else:
            pocketeer_file = f"{pocketeer_dir}/{uid}.cif_predictions.csv"
            prank_file = f"{prank_dir}/{uid}/{uid}.cif_predictions.csv"
            df_pocketeer, df_prank = pd.DataFrame(), pd.DataFrame()
            if os.path.exists(pocketeer_file):
                df_pocketeer = pd.read_csv(pocketeer_file, sep=r"\s*,\s*", engine="python")
            if os.path.exists(prank_file):
                df_prank = pd.read_csv(prank_file, sep=r"\s*,\s*", engine="python")

            if df_pocketeer.shape[0] > 0:
                resi_str = df_pocketeer["residue_ids"].values[0]
                residues = resi_str.split()
                res_dict[uid] = {"residues": residues, "comp_id": None}

            elif df_prank.shape[0] > 0:
                resi_str = df_prank["residue_ids"].values[0]
                residues = resi_str.split()
                res_dict[uid] = {"residues": residues, "comp_id": None}
    pkl_dump(save_file, res_dict)

if __name__ == "__main__":
    # get_afill_ligands(f"{root_path}/data/afilldb/", f"{root_path}/data/enzyme/RHEA/proc/afill_ligands.csv")  # 1

    get_similar_based_on_rxn_and_afill(
        f"{root_path}/data/enzyme/RHEA/proc/uid2rxn.pkl",
        f"{root_path}/data/enzyme/RHEA/proc/rxn2smi.json",
        f"{root_path}/data/enzyme/RHEA/proc/afill_ligands.csv",
        f"{root_path}/data/enzyme/RHEA/proc/comp_id2smi.json",
        f"{root_path}/data/enzyme/RHEA/proc/afill_sims.pkl",
    )  # 2

    batch_run_site(f"{root_path}/data/enzyme/RHEA/proc/afill_sims.pkl") # 3
    batch_run_prank(f"{root_path}/data/tmp/prank_teer", f"{root_path}/data/tmp/prank_pred")
    merge_all_sites(
        f"{root_path}/data/enzyme/RHEA/proc/uid2rxn.pkl",
        f"{root_path}/data/enzyme/RHEA/proc/sites_afill.pkl",
        f"{root_path}/data/tmp/prank_teer",
        f"{root_path}/data/tmp/prank_pred",
        f"{root_path}/data/enzyme/RHEA/proc/afill_ligands.csv",
        f"{root_path}/data/enzyme/RHEA/proc/final_sites.pkl",
    )
