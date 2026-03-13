import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tqdm import tqdm


def tranverse_folder(folder):
    filepath_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            filepath_list.append(os.path.join(root, file))
    return filepath_list


def run_quiet(cmd):
    # capture_output=True 会捕获输出而不打印到终端
    return subprocess.run(cmd, shell=True, capture_output=True)


def main_multiprocess(input_dir, pdb_fasta, pdb_redo_dir, n_blast=5, n_process=8):
    uid_list = []
    all_files = tranverse_folder(input_dir)
    for filename in all_files:
        if ".cif" in filename:
            uid_list.append(filename.split(".")[0])

    command_list = []
    for uid in tqdm(uid_list):
        input_structure_path = f"{uid}.cif"
        output_path = input_structure_path.replace("afdb", "afilldb")
        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(output_path):
            continue

        command = (
            f"alphafill process {input_structure_path} {output_path} --pdb-fasta={pdb_fasta} --pdb-dir={pdb_redo_dir} --blast-report-limit={n_blast}"
        )
        command_list.append(command)
    print(f"Total {len(command_list)} items to process.")

    with ThreadPoolExecutor(max_workers=n_process) as executor:
        list(tqdm(executor.map(run_quiet, command_list), total=len(command_list), desc="Processing items"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input directory containing CIF files", default=f"{root_path}/data/afdb")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for AlphaFill transplantion results", default=f"{root_path}/data/afilldb"
    )
    parser.add_argument("--pdb_fasta", type=str, default="/data/zzjun/datasets/pdb-redo.fa")
    parser.add_argument("--pdb_redo_dir", type=str, default="/data/zzjun/datasets/pdb-redo")
    args = parser.parse_args()

    main_multiprocess(args.input_dir, args.pdb_fasta, args.pdb_redo_dir, n_blast=250, n_process=32)
    # python run_afill.py --input_dir /data/zzjun/ECseek/data/afdb --output_dir /data/zzjun/ECseek/data/afilldb  --pdb_fasta /data/zzjun/datasets/pdb-redo_seqs.fa --pdb_redo_dir /data/zzjun/datasets/pdb-redo
