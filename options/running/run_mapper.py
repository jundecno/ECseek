import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from rdchiral.template_extractor import extract_from_reaction


def calc_rxnmapper_aam(data_path, save_dir, append=True, rerun=False):
    from rxnmapper import BatchedMapper

    save_path = os.path.join(save_dir, "rxn2aam_rxnmapper.pkl")
    if os.path.exists(save_path) and append and not rerun:
        cached_rxn2aam =pkl_load(save_path)
    else:
        cached_rxn2aam = {}

    rxn_mapper = BatchedMapper(batch_size=128)

    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data["CANO_RXN_SMILES"].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_rxn2aam]

    result_list = []
    for results in tqdm(rxn_mapper.map_reactions_with_info(rxns_to_run), total=len(rxns_to_run)):
        result_list.append(results.get("mapped_rxn"))

    rxn2aam = dict(zip(rxns_to_run, result_list))
    rxn2aam.update(cached_rxn2aam)

    pkl_dump(save_path, rxn2aam)


def calc_localmapper_aam(data_path, save_dir, append=True, rerun=False):
    from localmapper import localmapper

    mapper = localmapper(device="cpu")
    save_path = os.path.join(save_dir, "rxn2aam_localmapper.pkl")
    if os.path.exists(save_path) and append and not rerun:
        cached_rxn2aam = pkl_load(save_path)
    else:
        cached_rxn2aam = {}

    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data["CANO_RXN_SMILES"].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_rxn2aam]

    result_list = [mapper.get_atom_map(rxn) for rxn in tqdm(rxns_to_run)]

    rxn2aam = dict(zip(rxns_to_run, result_list))
    rxn2aam.update(cached_rxn2aam)

    pkl_dump(save_path, rxn2aam)


def rxn2template(rxn_smiles):
    reac, prod = rxn_smiles.split(">>")
    input_data = {"reactants": reac, "products": prod, "_id": "temp"}
    out = extract_from_reaction(input_data)
    return out["reaction_smarts"]  # type: ignore


def get_template(rxn_file, out_file):
    rxn_dict = pkl_load(rxn_file)
    rxn_res_dict = {}
    for rxn, aam in tqdm(rxn_dict.items()):
        try:
            template = rxn2template(aam)
        except Exception as e:
            print(f"Error processing reaction {rxn}: {e}")
            template = None
        rxn_res_dict[rxn] = template

    pkl_dump(out_file, rxn_res_dict)


def merge_rxnmapper_localmapper_aam(rxn2uid, rxn_mapper, local_mapper):
    df = pd.read_csv(rxn2uid)
    rxn2aam_rxnmapper = pkl_load(rxn_mapper)
    rxn2aam_localmapper = pkl_load(local_mapper)
    df["rxnmapper"] = df["CANO_RXN_SMILES"].map(rxn2aam_rxnmapper)
    df["localmapper"] = df["CANO_RXN_SMILES"].map(rxn2aam_localmapper)
    df["template"] = df["localmapper"].fillna(df["rxnmapper"])
    df = df.drop(columns=["rxnmapper", "localmapper"])
    df.to_csv(rxn2uid, index=False)


if __name__ == "__main__":
    data_path = f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv"
    save_dir = f"{root_path}/data/enzyme/RHEA/processed"
    # calc_rxnmapper_aam(data_path, save_dir)
    # calc_localmapper_aam(data_path, save_dir)
    # get_template(
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2aam_localmapper.pkl",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2template_localmapper.pkl",
    # )
    # get_template(
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2aam_rxnmapper.pkl",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2template_rxnmapper.pkl",
    # )
    merge_rxnmapper_localmapper_aam(
        f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
        f"{root_path}/data/enzyme/RHEA/processed/rxn2template_rxnmapper.pkl",
        f"{root_path}/data/enzyme/RHEA/processed/rxn2template_localmapper.pkl",
    )
