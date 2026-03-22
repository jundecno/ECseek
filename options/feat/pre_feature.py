import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer

def calc_seq_esm_C_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    model = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

    uid_to_seq = pkl_load(data_path)
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)
    uid_list = list(uid_to_seq.keys())
    print(f"\n{len(uid_list)} proteins to calculate features...")

    failed_uids = []
    mean_dict = pkl_load(esm_mean_feat_path) if os.path.exists(esm_mean_feat_path) else {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq[uid]
        save_path = os.path.join(esm_node_feat_dir, uid[:2], uid[2:4], uid[4:6], f"{uid}.npz")
        make_dir(os.path.dirname(save_path))

        protein = ESMProtein(sequence=seq)
        with torch.no_grad():
            protein_tensor = model.encode(protein)
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

        node_feature = logits_output.embeddings[0].cpu().numpy()  # type: ignore
        np.savez_compressed(save_path, node_feature=node_feature)
        mean_dict[uid] = node_feature.mean(axis=0)  # type: ignore

    pkl_dump(esm_mean_feat_path, mean_dict)
    print(f"\ncnt_fail: {len(failed_uids)}")


def calc_smiles_feature(rxn2smi):
    rxn2smi_dict = json_load(rxn2smi) # value是数组
    smis_set = set()
    for smis in rxn2smi_dict.values():
        smis_set.update(smis)
    print(len(smis_set))

    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    smi_feat_dict = {} if not os.path.exists(f"{root_path}/data/features/smi_feat.pkl") else pkl_load(f"{root_path}/data/features/smi_feat.pkl")
    for smis in tqdm(smis_set):
        if smis in smi_feat_dict:
            continue
        inputs = tokenizer([smis], padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        smi_feat_dict[smis] = outputs.pooler_output.cpu().numpy()
    pkl_dump(f"{root_path}/data/features/smi_feat.pkl", smi_feat_dict)

if __name__ == "__main__":
    # calc_seq_esm_C_feature(
    #     f"{root_path}/data/enzyme/RHEA/uid2seq.pkl", f"{root_path}/data/features/protein/", f"{root_path}/data/features/esm_mean_feat.pkl"
    # )
    calc_smiles_feature(f"{root_path}/data/enzyme/RHEA/proc/rxn2smi.json")
    # CUDA_VISIBLE_DEVICES=0 python options/feat/pre_feature.py
