import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *

# # # from rxnmapper import RXNMapper

# # # rxn_mapper = RXNMapper()
# # # rxns = [
# # #     "*C(=O)OCC.O>>*C(=O)O.CCO",
# # # ]
# # # results = rxn_mapper.get_attention_guided_atom_maps(rxns)
# # # print(results)
# # print(extract_reacting_center("*C(=O)OCC.O>>*C(=O)O.CCO",{"*C(=O)OCC.O>>*C(=O)O.CCO":"[*:1][C:2](=[O:3])[O:7][CH2:6][CH3:5].[OH2:4]>>[*:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]"}))

# # # import pickle as pkl
# # # data = pkl.load(open("/data/zzjun/ECseek/data/enzyme/ENZYME/uid2seq.pkl", "rb"))
# # # print(data["Q73GT3"])
# # # import pickle as pkl
# # # from rdchiral.template_extractor import extract_from_reaction
# # # data = pkl.load(open("/data/zzjun/ECseek/data/enzyme/RHEA/processed/rxn2template_localmapper.pkl", "rb"))
# # # # aam = list(data.values())[0]
# # # aam = data["CC(C)(O)C#N>>C#N.CC(C)=O", ]
# # # print(aam)
# # # reac, prod = aam.split(">>")
# # # input_data = {"reactants": reac, "products": prod,"_id": "test"}
# # # print(extract_from_reaction(input_data))


# # # import requests
# # # def get_fasta(uniprot_id):
# # #     url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
# # #     response = requests.get(url)

# # #     if response.status_code == 200:
# # #         return response.text
# # #     else:
# # #         return "未找到该 ID"


# # # # 示例
# # # print(get_fasta("Q9F7D8"))

# # # from esm.models.esmc import ESMC
# # # from esm.sdk.api import ESMProtein, LogitsConfig

# # # protein = ESMProtein(sequence="AAAAA")
# # # client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"
# # # protein_tensor = client.encode(protein)
# # # logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
# # # print(logits_output.logits, logits_output.embeddings)


# # # from esm.models.esm3 import ESM3
# # # from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


# # # # This will download the model weights and instantiate the model on your machine.
# # # model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu"

# # # # Generate a completion for a partial Carbonic Anhydrase (2vvb)
# # # prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
# # # protein = ESMProtein(sequence=prompt)
# # # # Generate the sequence, then the structure. This will iteratively unmask the sequence track.
# # # protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# # # # We can show the predicted structure for the generated sequence.
# # # protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
# # # protein.to_cif("./generation.cif")


# # # from rdkit import Chem
# # # from rdkit.Chem import AllChem

# # # smiles = "CCO"

# # # mol = Chem.MolFromSmiles(smiles)
# # # mol = Chem.AddHs(mol)

# # # # 生成3D构象
# # # AllChem.EmbedMolecule(mol)
# # # AllChem.UFFOptimizeMolecule(mol)

# # # # 计算体积
# # # vol = AllChem.ComputeMolVolume(mol)

# # # print("Molecular Volume:", vol)


# # from rdkit import Chem
# # from rdkit.rdBase import rdkitVersion
# # from rdkit.Chem import AllChem
# # from rdkit.Chem import Draw

# # # 1. 定义反应 SMARTS
# # rxn_smarts = "*/C(S)=N/OS(=O)(=O)O>>*C#N.O=S(=O)(O)O.S"

# # # 2. 从 SMARTS 创建反应对象
# # rxn = AllChem.ReactionFromSmarts(rxn_smarts)

# # # 3. 绘制反应方程式
# # # useSVG=True 可以生成更清晰的矢量图，False 则生成 PNG
# # img = Draw.ReactionToImage(rxn, subImgSize=(300, 300), useSVG=False)

# # # 4. 保存或显示
# # img.save("reaction.png")
# # data = pkl_load("/data/zzjun/ECseek/data/features/rxn_graph.pkl")
# # print(data["*/C(S)=N/OS(=O)(=O)O>>*C#N.O=S(=O)(O)O.S"])
# # print(pkl_load("/data/zzjun/ECseek/data/features/pocket_graph/A0/A0/09/A0A009IHW8.pkl"))

# # data = pkl_load("/data/zzjun/ECseek/data/features/pocket_graph/A0/A0/09/A0A009IHW8.pkl")
# # print(data)

# # from rxnfp.transformer_fingerprints import (
# #     RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
# # )

# # model, tokenizer = get_default_model_and_tokenizer()

# # rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

# # example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

# # fp = rxnfp_generator.convert(example_rxn)
# # print(len(fp))
# # print(fp[:5])

# import numpy as np
# from unimol_tools import UniMolRepr

# # single smiles unimol representation
# clf = UniMolRepr(
#     data_type="molecule",  # avaliable: molecule, oled, pocket. Only for unimolv1.
#     remove_hs=False,
#     model_name="unimolv1",  # avaliable: unimolv1, unimolv2
#     model_size="84m",  # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
# )
# smiles = "c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]"
# smiles_list = [smiles]
# unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
# # CLS token repr
# print(np.array(unimol_repr["cls_repr"]).shape)
# # atomic level repr, align with rdkit mol.GetAtoms()
# print(np.array(unimol_repr["atomic_reprs"]).shape)


# data = pkl_load("/data/zzjun/EnzySeek/data/training/clip_valid.pkl")
# print(type(data),data[:10])
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig

# protein = ESMProtein(sequence="AAAAA")
# client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
# print(logits_output.logits, logits_output.embeddings)
import torch

ckpt_path = "/data/zzjun/EnzySeek/checkpoints/new/2026-06-24_10-10-54/epoch_099.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(ckpt.keys())

for key, value in ckpt.items():
    if key == "state_dict":
        print(key, type(value))
