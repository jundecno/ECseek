import os
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def normalize_ec(ec_str: str) -> str:
    ec_items = ec_str.split(";")
    ec_level3 = set()

    for ec in ec_items:
        ec = ec.strip()
        if not ec.startswith("EC:"):
            continue

        parts = ec.replace("EC:", "").split(".")
        if len(parts) >= 3:
            ec_level3.add("EC:" + ".".join(parts[:3])+".*")

    return ";".join(sorted(ec_level3))


def tranverse_folder(folder):
    filepath_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            filepath_list.append(os.path.join(root, file))
    return filepath_list

def get_ligands(cif_file):
    doc = gemmi.cif.read(cif_file)
    block = doc.sole_block()
    if block.get_mmcif_category("_pdbx_entity_nonpoly"):
        return block.get_mmcif_category("_pdbx_entity_nonpoly")["comp_id"]
    else:
        return []


def pdb2cif(pdb_file, cif_file):
    structure = gemmi.read_pdb(pdb_file)
    doc = structure.make_mmcif_document()
    doc.write_file(cif_file)

def calc_tanimoto(smile1, smile2):
    m1 = Chem.MolFromSmiles(smile1)
    m2 = Chem.MolFromSmiles(smile2)

    if m1 is None or m2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def calculate_top_matrix(a, list_b):
    mol_a = Chem.MolFromSmiles(a)
    mols_b, res_b = [], []
    for s in list_b:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols_b.append(mol)
            res_b.append(s)

    fp_a = mfpgen.GetFingerprint(mol_a)
    fps_b = [mfpgen.GetFingerprint(m) for m in mols_b]
    sims = DataStructs.BulkTanimotoSimilarity(fp_a, fps_b)

    return a, res_b, np.array(sims)
