from matplotlib import pyplot as plt
import pandas as pd
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *


def cal_prot_counts():
    df = pd.read_csv(f"{root_path}/data/enzyme/ENZYME/enzyme.csv")
    lines = df["UniProt_IDs"].values
    prot_set = set()
    for i, line in enumerate(lines):
        if type(line) != str:
            print(f"Line {i} is not a string: {line}")
        prots = line.split(";")
        prot_set.update(prots)
    print(len(prot_set))

# 统计蛋白质数量
def statis_prots(fasta_file):
    prot_set = set()
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                prot_id = line.split()[0][1:]  # 去掉 '>'
                prot_set.add(prot_id)
    print(len(prot_set))


# 统计rxn中最大原子数的分布
def statis_max_atoms(rxn2smi_file):
    rxn2smi = json_load(rxn2smi_file)
    max_atoms_list = []
    for rxn_smiles, smis in rxn2smi.items():
        atom_nums = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                atom_nums.append(mol.GetNumAtoms())
        if len(atom_nums) > 0:
            max_atoms_list.append(max(atom_nums))
            if max(atom_nums) == 2:
                print(f"Reaction {rxn_smiles} has a substrate with {max(atom_nums)} atoms.")
                return
    # 统计max_atoms_list的分布
    # from collections import Counter
    # counter = Counter(max_atoms_list)
    # counter = counter.most_common()
    # json_dump(f"{root_path}/data/enzyme/RHEA/processed/rxn_max_atoms_distribution.json", dict(counter))

def plot_max_atoms_distribution(distribution_file):
    distribution = json_load(distribution_file)
    x = [int(item[0]) for item in distribution]
    y = [item[1] for item in distribution]
    plt.bar(x, y)
    plt.xlabel("Max Atoms in Substrate")
    plt.ylabel("Count")
    plt.title("Distribution of Max Atoms in Substrate for RHEA Reactions")
    plt.savefig(f"{root_path}/data/enzyme/RHEA/processed/rxn_max_atoms_distribution.png")

if __name__ == "__main__":
    # cal_prot_counts()
    # statis_prots(f"{root_path}/data/enzyme/ENZYME/af2_error.fasta")
    statis_max_atoms(f"{root_path}/data/enzyme/RHEA/processed/rxn2smi.json")
    # plot_max_atoms_distribution(f"{root_path}/data/enzyme/RHEA/processed/rxn_max_atoms_distribution.json")
    # data = json_load(f"{root_path}/data/enzyme/RHEA/processed/rxn_max_atoms_distribution.json")
    # keys = list(data.keys())
    # keys.sort(key=lambda x: int(x))
    # values = [data[key] for key in keys]
    # print(keys, values)
