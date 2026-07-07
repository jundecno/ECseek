import argparse

import rootutils


root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio.PDB import MMCIFParser, Select, MMCIFIO, PDBParser, PDBIO  # type: ignore


class PocketSelect(Select):
    def __init__(self, pocket_residues):
        self.pocket_residues = set(pocket_residues)

    def accept_residue(self, residue):  # type: ignore
        chain_resi = f"{residue.get_parent().id}_{residue.id[1]}"
        return chain_resi in self.pocket_residues


def run_pocketeer(args):
    pdb_file, tmp_dir = args
    base_name = os.path.basename(pdb_file).split(".")[0]
    if os.path.exists(f"{tmp_dir}/{base_name}_pocketeer/pockets.json"):
        return
    try:
        os.system(f"pocketeer {pdb_file} -o {tmp_dir}/{base_name}_pocketeer > /dev/null 2>&1")
    except:
        return


def get_poc_resi(args):
    file_path, pocket_residues, save_path, suffix = args
    if len(pocket_residues) == 0:
        return
    if suffix == "cif":
        cif_parser = MMCIFParser(QUIET=True)
        io = MMCIFIO()
        structure = cif_parser.get_structure("complex", file_path)[0]  # type: ignore
    else:
        cif_parser = PDBParser(QUIET=True)
        io = PDBIO()
        structure = cif_parser.get_structure("complex", file_path)[0]  # type: ignore
    io.set_structure(structure)
    io.save(f"{save_path}.{suffix}", PocketSelect(pocket_residues))


def pocket_detection_chunk(pdb_dir, out_dir):
    pocketeer_tmp_dir = f"{out_dir}/tmp/pocketeer"
    prank_teer_dir = f"{out_dir}/tmp/prank_teer"
    prank_pred_tmp_dir = f"{out_dir}/tmp/prank_pred"
    make_dir(pocketeer_tmp_dir)
    make_dir(prank_teer_dir)
    make_dir(prank_pred_tmp_dir)

    files = list(get_file_paths(pdb_dir))
    if len(files) == 0:
        raise ValueError("No files found in the specified directory.")
    suffix = files[0].split(".")[-1]
    pool = mlc.SuperPool(input_args.num_workers)
    # run pocketeer for those cif files
    results = pool.map(run_pocketeer, [(file, pocketeer_tmp_dir) for file in files], description="Running Pocketeer for pocket detection")

    # rescore with prank
    head_line = "PARAM.PREDICTION_METHOD=pocketeer\n\nHEADER: prediction protein\n\n"
    line_str = ""
    for file in os.listdir(pocketeer_tmp_dir):
        pdb_id = file.replace("_pocketeer", "")
        save_file = f"{pocketeer_tmp_dir}/{file}/pockets.json"
        if os.path.exists(save_file) and not os.path.exists(f"{prank_teer_dir}/{pdb_id}.{suffix}_predictions.csv"):
            line_str += f"{save_file} {pdb_dir}/{pdb_id}.{suffix}\n"

    # write ds
    print("Rescoring with Prank...")
    os.system(f"rm -f {out_dir}/tmp/pocketeer_prank.ds")
    write_txt(f"{out_dir}/tmp/pocketeer_prank.ds", head_line + line_str)
    os.system(f"{PRANK_BIN} rescore {out_dir}/tmp/pocketeer_prank.ds -c rescore_2024 -o {prank_teer_dir}  > /dev/null 2>&1")
    
    # only for those without predicted pockets from Pocketeer, run Prank predict
    cmd_strs = []
    for file in os.listdir(prank_teer_dir):
        if "_predictions.csv" in file:
            df = pd.read_csv(f"{prank_teer_dir}/{file}", sep=r"\s*,\s*", engine="python")
            pdb_id = file.split(".")[0]
            if df.shape[0] == 0 and not os.path.exists(f"{prank_pred_tmp_dir}/{pdb_id}/{pdb_id}.{suffix}_predictions.csv"):
                cmd_strs.append(f"{PRANK_BIN} predict -f {pdb_dir}/{pdb_id}.{suffix} -c rescore_2024 -o {prank_pred_tmp_dir}/{pdb_id} > /dev/null 2>&1")
    if len(cmd_strs) > 0:
        results = pool.map(os.system, cmd_strs, description="Running Prank for proteins without predicted pockets from Pocketeer")
    else:
        print("All pockets have been predicted by Pocketeer, no need to run Prank for those without predicted pockets from Pocketeer.")
    # extract pocket residues and save
    save_dir = f"{out_dir}/pocket"
    make_dir(save_dir)
    files = list(get_file_paths(pdb_dir))
    suffix = files[0].split(".")[-1]
    args_list = []
    for file in files:
        pdb_id = os.path.basename(file).split(".")[0]
        pocketeer_file = f"{prank_teer_dir}/{pdb_id}.{suffix}_predictions.csv"
        prank_file = f"{prank_pred_tmp_dir}/{pdb_id}/{pdb_id}.{suffix}_predictions.csv"
        if os.path.exists(f"{save_dir}/{pdb_id}.{suffix}"):
            continue
        df_pocketeer, df_prank = pd.DataFrame(), pd.DataFrame()
        if os.path.exists(pocketeer_file):
            df_pocketeer = pd.read_csv(pocketeer_file, sep=r"\s*,\s*", engine="python")
        elif os.path.exists(prank_file):
            df_prank = pd.read_csv(prank_file, sep=r"\s*,\s*", engine="python")

        target_residues = None
        if df_pocketeer.shape[0] > 0:
            target_residues = df_pocketeer["residue_ids"].values
        elif df_prank.shape[0] > 0:
            target_residues = df_prank["residue_ids"].values
        else:
            continue

        for i, residues in enumerate(target_residues):
            make_dir(f"{save_dir}/{pdb_id}")
            args_list.append((file, residues.split(), f"{save_dir}/{pdb_id}/pocket{i}", suffix))

    results = pool.map(get_poc_resi, args_list, description="Extracting pockets with Prank and Pocketeer")
    pool.exit()

if __name__ == "__main__":
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument("--pdb_dir", "-p", type=str, default="./data/pdb")
    parse_args.add_argument("--out_dir", "-o", type=str, default="./data/pocket")
    parse_args.add_argument("--num_workers", "-n", type=int, default=32)
    input_args = parse_args.parse_args()
    pocket_detection_chunk(input_args.pdb_dir, input_args.out_dir)
