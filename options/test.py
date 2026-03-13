import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))


# from rxnmapper import RXNMapper

# rxn_mapper = RXNMapper()
# rxns = [
#     "*OO.*N[C@@H](CS)C(*)=O.*N[C@@H](CS)C(*)=O>>*N[C@@H](CSSC[C@H](N*)C(*)=O)C(*)=O.*O.[H]O[H]",
# ]
# results = rxn_mapper.get_attention_guided_atom_maps(rxns)
# print(results)


# import pickle as pkl
# data = pkl.load(open("/data/zzjun/ECseek/data/enzyme/ENZYME/uid2seq.pkl", "rb"))
# print(data["Q73GT3"])
# import pickle as pkl
# from rdchiral.template_extractor import extract_from_reaction
# data = pkl.load(open("/data/zzjun/ECseek/data/enzyme/RHEA/processed/rxn2template_localmapper.pkl", "rb"))
# # aam = list(data.values())[0]
# aam = data["CC(C)(O)C#N>>C#N.CC(C)=O", ]
# print(aam)
# reac, prod = aam.split(">>")
# input_data = {"reactants": reac, "products": prod,"_id": "test"}
# print(extract_from_reaction(input_data))


# import requests
# def get_fasta(uniprot_id):
#     url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
#     response = requests.get(url)

#     if response.status_code == 200:
#         return response.text
#     else:
#         return "未找到该 ID"


# # 示例
# print(get_fasta("Q9F7D8"))

# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig

# protein = ESMProtein(sequence="AAAAA")
# client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
# print(logits_output.logits, logits_output.embeddings)


# from esm.models.esm3 import ESM3
# from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


# # This will download the model weights and instantiate the model on your machine.
# model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu"

# # Generate a completion for a partial Carbonic Anhydrase (2vvb)
# prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
# protein = ESMProtein(sequence=prompt)
# # Generate the sequence, then the structure. This will iteratively unmask the sequence track.
# protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# # We can show the predicted structure for the generated sequence.
# protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
# protein.to_cif("./generation.cif")


from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CCO"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# 生成3D构象
AllChem.EmbedMolecule(mol)
AllChem.UFFOptimizeMolecule(mol)

# 计算体积
vol = AllChem.ComputeMolVolume(mol)

print("Molecular Volume:", vol)
