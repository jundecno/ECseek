from multiprocessing import Pool
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
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


def calc_drfp(data_path, save_path, append=True):
    # 计算drfp指纹，去除环后计算
    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))
    if os.path.exists(save_path) and append:
        rxn_to_fp = pkl_load(save_path)
    else:
        rxn_to_fp = {}
    input_list = []

    for rxn in rxn_list:
        input_list.append(rxn)
        rxn_cano = cano_rxn(rxn, remove_stereo=True)
        if rxn_cano != rxn:
            input_list.append(rxn_cano)

    drfp_results = []
    with Pool(20) as p:
        for fp in tqdm(p.imap(DrfpEncoder.encode, input_list), total=len(input_list)):
            drfp_results.append(fp)
    for rxn, fp in zip(input_list, drfp_results):
        rxn_to_fp[rxn] = fp

    print(f"Number of reactions: {len(rxn_to_fp)}")
    check_dir(os.path.dirname(save_path))
    pkl_dump(save_path, rxn_to_fp)
    print(f"Save drfp to {save_path}")


if __name__ == "__main__":
    # calc_drfp(f"{root_path}/data/enzyme/ENZYME/processed_data.csv", f"{root_path}/data/enzyme/ENZYME/drfp.pkl")
    # /data/zzjun/ECseek/data/enzyme/features/drfp
    calc_drfp(f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv", f"{root_path}/data/enzyme/features/drfp/rhea.pkl")
