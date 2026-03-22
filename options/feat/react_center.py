import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *


def calc_reacting_center(data_path, aam_path, save_path, append=True):

    rxn2aam = pkl_load(aam_path)
    reacting_center_path = os.path.join(save_path)
    cached_reacting_center_map = pkl_load(reacting_center_path) if os.path.exists(reacting_center_path) and append else {}

    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data[RXN_COL].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_reacting_center_map]
    reacting_center_map = {}
    for rxn in tqdm(rxns_to_run):
        reacting_center_map[rxn] = extract_reacting_center(rxn, rxn2aam)

    if append:
        print(f"Append {len(reacting_center_map)} reacting center to {reacting_center_path}")

    reacting_center_map.update(cached_reacting_center_map)
    pkl_dump(reacting_center_path, reacting_center_map)

    if not append:
        print(f"Calculate {len(reacting_center_map)} reacting center and save to {reacting_center_path}")


if __name__ == "__main__":
    # calc_reacting_center(
    #     f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2aam_localmapper.pkl",
    #     f"{root_path}/data/enzyme/features/center/rxn2rc_localmapper.pkl",
    # )
    # calc_reacting_center(
    #     f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2aam_rxnmapper.pkl",
    #     f"{root_path}/data/enzyme/features/center/rxn2rc_rxnmapper.pkl",
    # )
    # calc_reacting_center(
    #     f"{root_path}/data/enzyme/RHEA/split/all_need.csv",
    #     f"{root_path}/data/enzyme/RHEA/proc/rxn2aam.pkl",
    #     f"{root_path}/data/features/rxn2rc.pkl",
    # )
    data = pkl_load("/data/zzjun/ECseek/data/features/drfp.pkl")
    print(len(data))
