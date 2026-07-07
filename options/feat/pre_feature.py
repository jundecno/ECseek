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


def calc_smiles_molformer_feature(rxn2smi):
    rxn2smi_dict = json_load(rxn2smi)  # value是数组
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


def calc_smiles_unimol_feature(rxn2smi, save_path):
    from unimol_tools import UniMolRepr

    rxn2smi_dict = json_load(rxn2smi)  # value是数组
    smis_set = set().union(*rxn2smi_dict.values())
    smiles_dict = {s: (s.replace("*", "C")) for s in smis_set}
    smiles_dict = {s: v for s, v in smiles_dict.items() if v != "[H][H]"}
    smis_idx_dict = json_load(f"{root_path}/data/enzyme/RHEA/proc/smi2idx.json")
    data = {"atoms": [], "coordinates": []}
    keys = []

    for key, value in smiles_dict.items():
        sdf_path = f"{root_path}/data/enzyme/RHEA/proc/sdf/{smis_idx_dict[value]}.sdf"
        if os.path.exists(sdf_path):
            mol = sdf_load(sdf_path)
            atom = [a.GetSymbol() for a in mol.GetAtoms()]
            coord = mol.GetConformer().GetPositions()
        else:
            mol = Chem.MolFromSmiles(value)
            atom = [a.GetSymbol() for a in mol.GetAtoms()]
            coord = np.zeros((len(atom), 3))

        keys.append(key)
        data["atoms"].append(atom)
        data["coordinates"].append(coord)

    clf = UniMolRepr(data_type="molecule", remove_hs=True, model_name="unimolv1", model_size="84m")
    unimol_repr = clf.get_repr(data, return_atomic_reprs=True)
    # CLS token repr
    cls_repr = unimol_repr["cls_repr"]
    atomic_repr = unimol_repr["atomic_reprs"]
    atomic_coords = unimol_repr["atomic_coords"]
    atomic_symbol = unimol_repr["atomic_symbol"]
    res_dict = {}
    for idx, key in enumerate(keys):
        res_dict[key] = {
            "cls_repr": cls_repr[idx],
            "atomic_repr": atomic_repr[idx],
            "atomic_coords": atomic_coords[idx],
            "atomic_symbol": atomic_symbol[idx],
        }
    # 单独给[H][H]创建嵌入
    h2_clf = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv1", model_size="84m")
    h2_repr = h2_clf.get_repr("[H][H]", return_atomic_reprs=True)
    res_dict["[H][H]"] = {
        "cls_repr": h2_repr["cls_repr"][0],
        "atomic_repr": h2_repr["atomic_reprs"][0],
        "atomic_coords": h2_repr["atomic_coords"][0],
        "atomic_symbol": h2_repr["atomic_symbol"][0],
    }
    pkl_dump(save_path, res_dict)


def calc_rxn_drfp_feature(data_path, save_path):
    from drfp import DrfpEncoder
    rxn_dict = json_load(data_path)

    rxn_to_fp = {}
    for rxn in tqdm(rxn_dict):
        rxn_to_fp[rxn] = DrfpEncoder.encode(clean_stereo(rxn))[0]

    pkl_dump(save_path, rxn_to_fp)


def calc_rxn_rxnfp_feature(data_path, save_path):
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)  # type: ignore

    rxn_dict = json_load(data_path)

    rxn_to_rxnfp = {}
    for rxn in tqdm(rxn_dict):
        rxn_to_rxnfp[rxn] = rxnfp_generator.convert(clean_stereo(rxn))  # type: ignore

    pkl_dump(save_path, rxn_to_rxnfp)


if __name__ == "__main__":
    # calc_seq_esm_C_feature(
    #     f"{root_path}/data/enzyme/RHEA/uid2seq.pkl", f"{root_path}/data/features/protein/", f"{root_path}/data/features/esm_mean_feat.pkl"
    # )
    calc_smiles_unimol_feature(f"{root_path}/data/enzyme/RHEA/proc/rxn2smi.json", f"{root_path}/data/features/unimol_feat.pkl")

    calc_rxn_drfp_feature(f"{root_path}/data/features/rxn2normal.json", f"{root_path}/data/features/rxn_drfp.pkl")
    calc_rxn_rxnfp_feature(f"{root_path}/data/features/rxn2normal.json", f"{root_path}/data/features/rxn_rxnfp.pkl")

