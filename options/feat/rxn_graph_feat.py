import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))

from utils import *

from multiprocessing import Pool
from drfp import DrfpEncoder


def cano_smiles(smiles, remove_stereo=False):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def cano_rxn(rxn, exchange_pos=False, remove_stereo=False):
    data = rxn.split(">")
    reactants = data[0].split(".")
    reactants = [cano_smiles(each, remove_stereo) for each in reactants]
    products = data[-1].split(".")
    products = [cano_smiles(each, remove_stereo) for each in products]
    reactants = sorted(reactants)  # type: ignore
    products = sorted(products)  # type: ignore
    if exchange_pos:
        new_rxn = f"{'.'.join(products)}>>{'.'.join(reactants)}"
    else:
        new_rxn = f"{'.'.join(reactants)}>>{'.'.join(products)}"
    return new_rxn


def wapper(rxn):
    return rxn, DrfpEncoder.encode(rxn)


def calc_drfp(data_path, save_path):
    # 计算drfp指纹，去除环后计算
    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))

    pool = mlc.SuperPool(20)
    results = pool.map(wapper, rxn_list, 32)

    rxn_to_fp = {}
    for rxn, fp in results:
        if fp is not None:
            rxn_to_fp[rxn] = fp[0]

    print(f"Number of reactions: {len(rxn_to_fp)}")
    check_dir(os.path.dirname(save_path))
    pkl_dump(save_path, rxn_to_fp)
    print(f"Save drfp to {save_path}")


def cal_rxnfp(data_path, save_path):
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)  # type: ignore

    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))
    rxn_to_rxnfp = {}
    batch_size = 32
    for i in tqdm(range(0, len(rxn_list), batch_size)):
        batch_rxn = rxn_list[i : i + batch_size]
        batch_rxnfp = rxnfp_generator.convert_batch(batch_rxn)  # type: ignore
        for rxn, fp in zip(batch_rxn, batch_rxnfp):
            rxn_to_rxnfp[rxn] = fp
    print(f"Number of reactions: {len(rxn_to_rxnfp)}")
    check_dir(os.path.dirname(save_path))
    pkl_dump(save_path, rxn_to_rxnfp)
    print(f"Save rxnfp to {save_path}")


if __name__ == "__main__":
    # /data/zzjun/ECseek/data/enzyme/features/drfp
    calc_drfp(f"{root_path}/data/enzyme/RHEA/split/all_need.csv", f"{root_path}/data/features/drfp.pkl")
    # cal_rxnfp(f"{root_path}/data/enzyme/RHEA/split/all_need.csv", f"{root_path}/data/features/rxnfp.pkl")
