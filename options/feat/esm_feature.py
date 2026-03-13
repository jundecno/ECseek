import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def calc_seq_esm_C_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    model = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

    df_data = pd.read_csv(data_path)
    uid_to_seq = dict(zip(df_data[UID_COL], df_data[SEQ_COL]))
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)
    uid_list = list(uid_to_seq.keys())
    print(f"\n{len(uid_list)} proteins to calculate features...")

    failed_seqs = []
    failed_uids = []
    seq_to_feature = pkl_load(esm_mean_feat_path) if os.path.exists(esm_mean_feat_path) else {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq[uid]
        save_path = os.path.join(esm_node_feat_dir, uid[:2], uid[2:4], uid[4:6], f"{uid}.npz")
        make_dir(os.path.dirname(save_path))
        if os.path.exists(save_path):
            continue
        protein = ESMProtein(sequence=seq)
        try:
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        except Exception as e:
            print(f"sequence length: {len(seq)}")
            failed_seqs.append(seq)
            failed_uids.append(uid)
            continue

        node_feature = logits_output.embeddings[0].cpu().numpy()  # type: ignore

        np.savez_compressed(save_path, node_feature=node_feature)

        seq_to_feature[seq] = node_feature.mean(axis=0)  # type: ignore

    pkl_dump(esm_mean_feat_path, seq_to_feature)
    print(f"\ncnt_fail: {len(failed_uids)}")
    df_failed = pd.DataFrame({UID_COL: failed_uids, SEQ_COL: failed_seqs})
    failed_save_path = os.path.join(esm_node_feat_dir, "failed_proteins.csv")
    df_failed.to_csv(failed_save_path, index=False)
    print(f"Save failed proteins to {failed_save_path}")

if __name__ == "__main__":
    calc_seq_esm_C_feature(
        f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
        f"{root_path}/data/enzyme/features/protein/",
        f"{root_path}/data/enzyme/features/protein/esm_mean_feat.pkl",
    )
