import argparse
import glob

from regex import F
import rootutils


root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *


def run_drugclip(pocket_dir, save_dir):
    # run drugclip for those cif files
    make_dir(save_dir)
    pool = mlc.SuperPool(3)
    cmd_list = []
    for file_dir in tqdm(os.listdir(pocket_dir)):
        if not os.path.exists(f"{save_dir}/{file_dir}.txt"):
            cmd_list.append(
                f"bash /data01/zzjun/ECseek/tools/Drug-The-Whole-Genome/retrieval.sh {pocket_dir}/{file_dir}/pocket.lmdb {save_dir}/{file_dir}.txt"
            )
    pool.map(os.system, cmd_list, description="Running drugclip for pockets")


def find_steroid(drugclip_result_dir):
    steroid_dict = pkl_load(f"{root_path}/help_data/steroid.pkl")
    hit_dict = pkl_load(f"{root_path}/help_data/hit.pkl")

    keys = sorted(list(steroid_dict.keys()))
    values = [steroid_dict[key] for key in keys]
    res_dict = {}
    pool = mlc.SuperPool(32)
    for file_path in tqdm(glob.glob(f"{drugclip_result_dir}/*.txt")):
        pdb_id = os.path.basename(file_path).split(".")[0]
        res_dict[pdb_id] = {"ligand": [], "max_sim": [], "hit_id": []}
        hit_id_list = []
        hit_fp_list = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(",")
                hit_id_list.append(items[0])
                hit_fp_list.append((hit_dict[items[0]], values))

        results = pool.map(calc_top_fp_sim, hit_fp_list, description=f"Calculating Tanimoto similarity")
        for idx, result in enumerate(results):
            max_idx, max_sim = result
            if max_sim > 0.8:
                lig_name = keys[max_idx]
                res_dict[pdb_id]["hit_id"].append(hit_id_list[idx])
                res_dict[pdb_id]["ligand"].append(lig_name)
                res_dict[pdb_id]["max_sim"].append(max_sim)
    # save results for each pdb_id
    # save as csv
    pkl_dump(f"{root_path}/help_data/Rg_results/steroid_hit.pkl", res_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-dir", "-p", type=str, default="", help="path for pocket dir")
    parser.add_argument("--item-name", "-i", type=str, default="", help="item name for drugclip")
    parser.add_argument("--save-dir", "-s", type=str, default="", help="path for saving drugclip results")
    parser.add_argument("--result-dir", "-r", type=str, default="", help="path for drugclip results")
    args = parser.parse_args()
    # run_drugclip(args.pocket_dir, args.save_dir)
    find_steroid(args.result_dir)
