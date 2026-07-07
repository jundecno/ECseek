import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from rdchiral.template_extractor import extract_from_reaction


def calc_rxnmapper_aam(data_path, save_dir):
    from rxnmapper import BatchedMapper
    save_path = os.path.join(save_dir, "rxn2aam_rxnmapper.pkl")

    rxn_mapper = BatchedMapper(batch_size=128)

    rxn_dict = json_load(data_path)
    rxn_values = list(rxn_dict.values())

    mapper_list = []
    for results in tqdm(rxn_mapper.map_reactions_with_info(rxn_values), total=len(rxn_values)):
        mapper_list.append(results.get("mapped_rxn"))

    rxn2aam = dict(zip(rxn_values, mapper_list))

    pkl_dump(save_path, rxn2aam)


def calc_localmapper_aam(data_path, save_dir):
    from localmapper import localmapper

    mapper = localmapper(device="cuda")
    save_path = os.path.join(save_dir, "rxn2aam_localmapper.pkl")

    rxn_dict = json_load(data_path)
    rxn_values = list(rxn_dict.values())

    mapper_list = [mapper.get_atom_map(rxn) for rxn in tqdm(rxn_values)]
    rxn2aam = dict(zip(rxn_values, mapper_list))

    pkl_dump(save_path, rxn2aam)


def rxn2template(rxn_smiles):
    reac, prod = rxn_smiles.split(">>")
    input_data = {"reactants": reac, "products": prod, "_id": "temp"}
    out = extract_from_reaction(input_data)
    return out["reaction_smarts"]  # type: ignore


def get_template(rxn_file, out_file):
    rxn_dict = json_load(rxn_file)
    rxn_res_dict = {}
    for rxn, aam in tqdm(rxn_dict.items()):
        try:
            template = rxn2template(aam)
        except Exception as e:
            print(f"Error processing reaction {rxn}: {e}")
            template = None
        rxn_res_dict[rxn] = template

    json_dump(out_file, rxn_res_dict)


def merge_rxnmapper_localmapper_aam(rxn_mapper, local_mapper):
    rxn2aam_rxnmapper = pkl_load(rxn_mapper)
    rxn2aam_localmapper = pkl_load(local_mapper)
    merged_dict = {}
    for rxn in rxn2aam_rxnmapper:
        value = rxn2aam_rxnmapper[rxn]
        if value == None:
            rxn2aam_rxnmapper[rxn] = rxn2aam_localmapper[rxn]
    merged_dict = rxn2aam_rxnmapper
    json_dump(os.path.join(save_dir, "rxn2aam.json"), merged_dict)


if __name__ == "__main__":
    data_path = f"{root_path}/data/features/rxn2normal.json"
    save_dir = f"{root_path}/data/features"
    calc_rxnmapper_aam(data_path, save_dir)
    calc_localmapper_aam(data_path, save_dir)
    merge_rxnmapper_localmapper_aam(
        f"{root_path}/data/features/rxn2aam_rxnmapper.pkl",
        f"{root_path}/data/features/rxn2aam_localmapper.pkl",
    )
    # get_template(
    #     f"{root_path}/data/features/rxn2aam.json",
    #     f"{root_path}/data/features/rxn2temp.json",
    # )
    # merge_rxnmapper_localmapper_aam(
    #     f"{root_path}/data/enzyme/RHEA/processed/rhea_rxn2uids.csv",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2template_rxnmapper.pkl",
    #     f"{root_path}/data/enzyme/RHEA/processed/rxn2template_localmapper.pkl",
    # )
    # data = json_load(f"{root_path}/data/features/rxn_cleaned.json")
    # print(len(set(data.values())))
