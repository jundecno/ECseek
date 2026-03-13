# 使用pocketeer预测后，使用p2rank进行结构评估，选取top1的结构进行后续分析
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import rootutils


root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import ast
from rdkit.Chem import rdMolDescriptors


def get_afill_error(afill_dir, save_file):
    # 如果AlphaFill没有填充任何配体, 则说明该蛋白质没有已知的配体结合位点, 可能是因为该蛋白质没有已知的配体结合位点, 或者AlphaFill没有找到任何匹配的模板结构.
    uid_list = []
    all_files = tranverse_folder(afill_dir)

    for filename in all_files:
        if ".cif" in filename:
            uid_list.append(filename.split(".")[0])

    ligand_dict = {}
    for uid in tqdm(uid_list):
        file_id = os.path.basename(uid)
        ligands = get_ligands(f"{uid}.cif")
        ligand_dict[file_id] = ligands
        if len(ligands) == 0:
            append_txt(f"{root_path}/options/running/afill_err.txt", f"{file_id}\n")
    # save csv file
    pd.DataFrame({"uid": list(ligand_dict.keys()), "ligands": list(ligand_dict.values())}).to_csv(save_file, index=False)


def get_similar_based_on_rxn_and_afill(rxn2uid, rxn2smi, afill_ligands_cav, comp_id2smi, save_file):
    df_data = pd.read_csv(rxn2uid)
    rxn_smis = json_load(rxn2smi)  # for smiles from rxn
    afill_ligands = pd.read_csv(afill_ligands_cav, converters={"ligands": ast.literal_eval})
    afill_ligands = dict(zip(afill_ligands["uid"].values, afill_ligands["ligands"].values))  # for rcsb
    comp_id2smi_dict = json_load(comp_id2smi)  # for smiles from comp_id
    res_dict = {}

    for idx, row in tqdm(df_data.iterrows()):
        uid = row[UID_COL]
        rxn_smiles = row[RXN_COL]
        smis_from_rxn = rxn_smis[rxn_smiles]

        # 使用原子数目最大的底物进行相似度计算
        atom_max = -1
        rxn_smiles_best = ""
        for rxn_smiles_str in smis_from_rxn:
            mol = Chem.MolFromSmiles(rxn_smiles_str)
            if mol is None:
                continue
            atom_num = mol.GetNumAtoms()
            if atom_num > atom_max:
                atom_max = atom_num
                rxn_smiles_best = rxn_smiles_str

        if atom_max == -1:
            res_dict[(uid, rxn_smiles)] = {"smiles_rxn": "", "smiles_afill": [], "sims": np.array([])}
            continue
        # 计算rxn_smiles_best与afill_ligands中所有ligand smiles的相似度，找到最相似的ligand smiles
        smis_from_afill = [comp_id2smi_dict[comp_id] for comp_id in afill_ligands.get(uid, [])]
        smis_sims = calculate_top_matrix(rxn_smiles_best, smis_from_afill)
        res_dict[(uid, rxn_smiles)] = {"smiles_rxn": smis_sims[0], "smiles_afill": smis_sims[1], "sims": smis_sims[2]}
    pkl_dump(save_file, res_dict)


def get_max_atoms_idx(smi_list):
    max_atoms = -1
    max_idx = -1
    for idx, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atom_num = mol.GetNumAtoms()
        if atom_num > max_atoms:
            max_atoms = atom_num
            max_idx = idx
    return max_atoms, max_idx


def run_pocketeer(cif_file, tmp_dir):
    base_name = os.path.basename(cif_file).split(".")[0]
    if os.path.exists(f"{tmp_dir}/{base_name}_pocketeer/pockets.json"):
        return
    try:
        os.system(f"pocketeer {cif_file} -o {tmp_dir}/{base_name}_pocketeer > /dev/null 2>&1")
    except:
        return


def batch_run_site(rxn_afill_ligands):
    rxn_afill_ligands_dict = pkl_load(rxn_afill_ligands)
    tmp_dir = f"{root_path}/options/running/pocketeer"
    res_afill_dict = {}
    cif_files = []
    for (uid, rxn_smiles), data in tqdm(rxn_afill_ligands_dict.items()):
        # 如果data中没有大于0.5的相似度，则说明该反应的底物与afill的配体没有相似的分子，此时可以选择跳过该反应，或者选择相似度最高的那个配体进行后续分析
        rxn_smi = data["smiles_rxn"]
        afill_smi = data["smiles_afill"]
        sims = data["sims"]

        if sims.size == 0 or sims.max() < 0.5:
            max_atoms, max_idx = get_max_atoms_idx(afill_smi)
            if max_atoms > 10:
                # 仍使用alphafill的配体进行预测
                afill_smi_best = afill_smi[max_idx]
                res_afill_dict[(uid, rxn_smiles)] = {"smiles_rxn": rxn_smi, "smiles_afill": afill_smi_best}
            else:
                # 使用prank进行预测
                cif_file = f"{AFDB}/{uid2path(uid)}"
                cif_files.append(cif_file)
        else:
            # 确定alphafill填充的配体
            afill_smi_best = afill_smi[sims.argmax()]
            res_afill_dict[(uid, rxn_smiles)] = {"smiles_rxn": rxn_smi, "smiles_afill": afill_smi_best}
    # run pocketeer for those cif files
    with Pool(processes=16) as pool:
        results = pool.starmap(run_pocketeer, [(cif_file, tmp_dir) for cif_file in cif_files])

    line_str = ""
    for file in os.listdir(tmp_dir):
        uid = file.split(".")[0]
        save_file = f"{tmp_dir}/{uid}/pockets.json"
        cif_file = f"{AFDB}/{uid2path(uid)}".replace("_pocketeer", "")
        if os.path.exists(save_file):
            line_str += f"{save_file} {cif_file}\n"

    pkl_dump(f"{root_path}/data/enzyme/RHEA/processed/sites_afill.pkl", res_afill_dict)
    # write ds
    ds_str = "PARAM.PREDICTION_METHOD=pocketeer\n\nHEADER: prediction protein\n\n" + line_str
    write_txt(f"{root_path}/data/enzyme/RHEA/processed/pocketeer_prank.ds", ds_str)
    os.system(
        f"{PRANK_BIN} rescore {root_path}/data/enzyme/RHEA/processed/pocketeer_prank.ds -c rescore_2024 -o {tmp_dir.replace("pocketeer", "prank")}"
    )


def batch_run_prank(prank_path, save_path):
    # 如果pocketeer没有预测出结合位点，则使用prank进行预测
    cmd_strs = []
    for file in os.listdir(prank_path):
        if ".csv" in file:
            df = pd.read_csv(f"{prank_path}/{file}", sep=r"\s*,\s*", engine="python")
            uid = file.split(".")[0]
            # 如果pocketeer没有预测出结合位点，则说明该蛋白质没有已知的配体结合位点, 可能是因为该蛋白质没有已知的配体结合位点, 或者pocketeer没有找到任何匹配的模板结构.
            if df.shape[0] == 0:
                cmd_strs.append(f"{PRANK_BIN} predict -f {AFDB}/{uid2path(uid)} -o {save_path}/{uid} > /dev/null 2>&1")
    with Pool(processes=16) as pool:
        results = pool.starmap(os.system, [(cmd_str,) for cmd_str in cmd_strs])


def merge_all_sites(rxn2uid, afill_sites, pocketeer_dir, prank_dir, afill_ligands, save_file):
    # 确定逻辑：优先选择afill，否则选择pocketeer，如果pocketeer没有预测出结合位点，则选择prank，最后使用afill填充的配体
    # 确定最终的结合位点文件格式，包含uid, rxn_smiles, residues, compid
    rxn2uid_df = pd.read_csv(rxn2uid)
    afill_sites_dict = pkl_load(afill_sites)
    afill_ligands_dict = pd.read_csv(afill_ligands, converters={"ligands": ast.literal_eval})
    afill_ligands_dict = dict(zip(afill_ligands_dict["uid"].values, afill_ligands_dict["ligands"].values))
    compid_dict = json_load(f"{root_path}/data/enzyme/RHEA/processed/comp_id2smi.json")
    compid_reverse_dict = {v: k for k, v in compid_dict.items()}

    res_dict = {}
    for idx, row in tqdm(rxn2uid_df.iterrows()):
        uid = row[UID_COL]
        rxn_smiles = row[RXN_COL]
        if (uid, rxn_smiles) in afill_sites_dict:
            ligand_smiles = afill_sites_dict[(uid, rxn_smiles)]["smiles_afill"]
            comp_id = compid_reverse_dict[ligand_smiles]
            res_dict[(uid, rxn_smiles)] = {"residues": None, "comp_id": comp_id}
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
                res_dict[(uid, rxn_smiles)] = {"residues": residues, "comp_id": None}

            elif df_prank.shape[0] > 0:
                resi_str = df_prank["residue_ids"].values[0]
                residues = resi_str.split()
                res_dict[(uid, rxn_smiles)] = {"residues": residues, "comp_id": None}
            else:
                afill_compids = afill_ligands_dict.get(uid, [])
                if len(afill_compids) == 0:
                    res_dict[(uid, rxn_smiles)] = {"residues": None, "comp_id": None}
                    continue
                afill_smis = [compid_dict[comp_id] for comp_id in afill_compids]
                max_atoms, max_idx = get_max_atoms_idx(afill_smis)
                max_smiles = afill_smis[max_idx]
                comp_id = compid_reverse_dict[max_smiles]
                res_dict[(uid, rxn_smiles)] = {"residues": None, "comp_id": comp_id}
    pkl_dump(save_file, res_dict)

if __name__ == "__main__":
    # get_afill_error(f"{root_path}/data/afilldb/", f"{root_path}/data/enzyme/RHEA/processed/afill_ligands.csv") # 1

    # get_similar_based_on_rxn_and_afill(
    #     f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2smi.json",
    #     f"{root_path}/data/enzyme/RHEA/processed/afill_ligands.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/comp_id2smi.json",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2afill_ligands.pkl",
    # ) # 2

    # batch_run_site(f"{root_path}/data/enzyme/RHEA/processed/rxn2afill_ligands.pkl") # 3
    # batch_run_prank(f"{root_path}/options/running/prank_teer", f"{root_path}/options/running/prank_pred")
    # merge_all_sites(
    #     f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/sites_afill.pkl",
    #     f"{root_path}/options/running/prank_teer",
    #     f"{root_path}/options/running/prank_pred",
    #     f"{root_path}/data/enzyme/RHEA/processed/afill_ligands.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/final_sites.pkl",
    # )
    # sdf_file = "/data/zzjun/ECseek/data/enzyme/RHEA/processed/sdf/1000.sdf"
    # mol = Chem.SDMolSupplier(sdf_file)[0]
    # # print(Chem.Descriptors3D.MolVolume(mol))
    # print(Chem.MolToSmiles(mol))
    # ['Q1J288', 'B1I4L8', 'Q24VC8', 'B7MH66', 'B7M603', 'A5EW83']
    data = pkl_load(f"{root_path}/data/enzyme/RHEA/processed/rxn2afill_ligands.pkl")
    # 打印前10条数据
    print(len(data))
    df = pd.read_csv(f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv")
    res_dict = {}
    for idx, row in df.iterrows():
        uid = row[UID_COL]
        rxn_smiles = row[RXN_COL]
        res_dict[(uid, rxn_smiles)] = 0
        if uid in set(['Q1J288', 'B1I4L8', 'Q24VC8', 'B7MH66', 'B7M603', 'A5EW83']):
            print(uid)
    print(len(res_dict))
