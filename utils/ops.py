import os
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from Bio.PDB import PDBParser, MMCIFIO, MMCIFParser # type: ignore
import mlcrate as mlc
import torch
from torch_geometric.utils import to_undirected, add_self_loops

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

def pdb2cif(pdb_file, cif_file):
    # structure = gemmi.read_pdb(pdb_file)
    # doc = structure.make_mmcif_document()
    # doc.write_file(cif_file)
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_file)
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_file)

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


def get_resi_len(cif_file):
    uid = os.path.basename(cif_file).split(".")[0]
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)
    model = structure[0] # TYPE:IGNORE
    return uid, len(list(model.get_residues()))


def normalize_smiles(input: str, canonical=True, isomeric=False):
    if input.endswith(".mol2"):
        mol = Chem.MolFromMol2File(input)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
        else:
            unstand_mol = Chem.MolFromMol2File(input, sanitize=False, cleanupSubstructures=False)
            mol = Chem.RemoveHs(unstand_mol, sanitize=False)
            return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    else:
        return Chem.MolToSmiles(Chem.MolFromSmiles(input), canonical=canonical, isomericSmiles=isomeric)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = [False] * len(allowable_set)
    return list(map(lambda s: x == s, allowable_set))


def make_undirected_with_self_loops(edge_index, edge_attr: torch.Tensor, undirected=True, self_loops=True):
    if undirected:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    if self_loops:
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0)
    return edge_index, edge_attr

bonds_allowed = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
