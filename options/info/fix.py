from re import split

import Bio
import Bio.Seq
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio import SeqIO

def fix_rxn(train_csv, valid_csv, rxn2rc, rxn2smi):
    rxn_rc_dict = pkl_load(rxn2rc)
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    all_df = set(train_df[RXN_COL].unique()) | set(valid_df[RXN_COL].unique())
    missing_rxns = set()
    for rxn in tqdm(all_df):
        if rxn not in rxn_rc_dict:
            missing_rxns.add(rxn)
    print(f"Missing reactions in rxn2rc mapping: {missing_rxns}")
    rxn2smi_dict = json_load(rxn2smi)
    for rxn in missing_rxns:
        if rxn in rxn2smi_dict:
            print(f"Reaction {rxn} is missing in rxn2rc but has SMILES: {rxn2smi_dict[rxn]}")
        else:
            smiles = rxn.replace(">>", ".").split(".")
            print(smiles)
            return

def fix_prot(uid2seq_path, err_fasta_path):
    uid2seq = pkl_load(uid2seq_path)
    for uid in uid2seq:
        if uid == "Q8NEZ4":
            append_txt("tmp.fa", f">{uid}\n{uid2seq[uid]}\n")
        
    # err_fa_dict = SeqIO.to_dict(SeqIO.parse(err_fasta_path, "fasta"))
    # for uid_desc in err_fa_dict:
    #     uid = uid_desc.split("|")[1]
    #     if uid in uid2seq:
    #         print(f"UID {uid} is in both uid2seq and err_fasta. Original length: {len(uid2seq[uid])}, Error seq: {err_fa_dict[uid_desc].seq}")
    #         uid2seq[uid] = str(err_fa_dict[uid_desc].seq)
    #     else:
    #         print(f"UID {uid} is in err_fasta but not in uid2seq.")
    # pkl_dump(uid2seq_path, uid2seq)

if __name__ == "__main__":
    # fix_rxn(
    #     f"{root_path}/data/training/train.csv",
    #     f"{root_path}/data/training/valid.csv",
    #     f"{root_path}/data/features/center/rxn2rc_localmapper.pkl",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2smi.json",
    # )
    fix_prot(f"{root_path}/data/enzyme/RHEA/uid2seq.pkl", "/data/zzjun/ECseek/options/data/err.fa")
