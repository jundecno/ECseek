import copy
from numpy import save
from regex import R
import rootutils
import torch

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio import SeqIO
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# This will download the model weights and instantiate the model on your machine.


def run_esm3(fasta_file, save_dir):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    # 根据序列长度排序，优先处理较短的序列，以减少内存占用和加速处理速度
    records.sort(key=lambda r: len(r.seq))
    print(len(records[0].seq), len(records[-1].seq))
    
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open",device=torch.device("cuda")) # or "cpu"
    for record in tqdm(records):
        uid = record.id
        sequence = str(record.seq)
        if os.path.exists(os.path.join(save_dir, f"{uid}.pdb")):
            continue
        try:
            protein = ESMProtein(sequence=sequence)
            protein = model.generate(
                protein,
                GenerationConfig(track="structure", num_steps=30, temperature=0.7),
            )
            output_path = os.path.join(save_dir, f"{uid}.pdb")
            protein.to_pdb(output_path) # type: ignore
        except Exception as e:
            print(f"Error processing {uid}: {e}")


def copy_files_to_afdb(src_dir):
    all_files = tranverse_folder(src_dir)
    for file in tqdm(all_files):
        save_file = os.path.join(AFDB, uid2path(os.path.basename(file).split(".")[0]))
        pdb2cif(file, save_file)

if __name__ == "__main__":
    # run_esm3(fasta_file="/data/zzjun/ECseek/data/enzyme/RHEA/af2_err.fa", save_dir=f"{root_path}/data/esmfold")
    # copy_files_to_afdb(src_dir=f"{root_path}/data/esmfold/")
    fa_dict = read_fasta("/data/zzjun/ECseek/data/enzyme/RHEA/af2_err.fa")
    for uid, seq in fa_dict.items():
        # os.system(f"rm {AFILLDB}/{uid2path(uid)}")
        print(f"{AFDB}/{uid2path(uid)}")
        print(f"{AFILLDB}/{uid2path(uid)}")
