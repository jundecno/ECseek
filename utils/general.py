import os
import pickle as pkl
import json
import pandas as pd
import numpy as np
import re
from Bio import PDB
import periodictable
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SeqUtils import seq1
from rdkit import Chem

# general config
parser = PDB.PDBParser(QUIET=True)

# easy function
elem2num = lambda x: periodictable.elements.symbol(x[:1] + x[1:].lower()).number
elem2mass = lambda x: periodictable.elements.symbol(x[:1] + x[1:].lower()).mass


def pkl_load(file_path):
    """
    读取文件内容
    """
    with open(file_path, "rb") as file:
        return pkl.load(file)


def pkl_dump(file_path, content):
    """
    写内容到文件
    """
    with open(file_path, "wb") as file:
        pkl.dump(content, file)


def json_load(file_path):
    """
    读取文件内容
    """
    with open(file_path, "r") as file:
        return json.load(file)


def json_dump(file_path, content):
    """
    写内容到文件
    """
    with open(file_path, "w") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)


def append_txt(file_path, content):
    """
    追加内容到文件
    """
    with open(file_path, "a") as file:
        file.write(content)


def write_txt(file_path, content):
    """
    写内容到文件
    """
    with open(file_path, "w") as file:
        file.write(content)


def read_file(file_path):
    """
    读取文件内容
    """
    with open(file_path, "r") as file:
        return file.read()


def read_lines(file_path):
    """
    读取文件内容
    """
    with open(file_path, "r") as file:
        res = file.readlines()
    return [x.strip() for x in res]


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)  # 如果存在，则不报错


def pdb2fasta(pdb_file, protein_name=None):
    if protein_name is None:
        protein_name = os.path.basename(pdb_file).split(".")[0]
    protein = parser.get_structure("protein", pdb_file)[0]
    res_str = ""
    for chain in protein:
        chain_str = f">{protein_name}_{chain.id}\n"
        for residue in chain:
            res_name = residue.get_resname()
            # 缩写
            chain_str += protein_letters_3to1_extended[res_name] if res_name in protein_letters_3to1_extended else "X"
        res_str += chain_str + "\n"
    return res_str


def pdb2dict(pdb_file):
    protein = parser.get_structure("protein", pdb_file)[0]  # type: ignore
    res_dict = {}
    order_list = []
    for chain in protein:
        aa_resnames = []
        for residue in chain:
            res_id = residue.get_id()
            resname = residue.resname.strip()
            if res_id[0] == " ":
                aa_resnames.append(resname)
        if len(aa_resnames) > 0:
            res_dict[chain.id] = seq1("".join(aa_resnames))
            order_list.append(chain.id)
    res_dict["order"] = order_list
    return res_dict


def merge_seq(dest, target):
    dest = list(dest)
    target = list(target)
    return "".join([str(int(x) or int(y)) for x, y in zip(dest, target)])


def read_fasta(path, label=False, skew=0):
    res_dict = {}
    if label:
        with open(path, "r") as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[:: 2 + skew]:
                name = content[idx].replace(">", "").replace("\n", "").strip()
                seq = content[idx + 1 + skew].replace("\n", "")
                if name in res_dict.keys():
                    res_dict[name] = merge_seq(res_dict[name], seq)
                else:
                    res_dict[name] = seq
    else:
        with open(path, "r") as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[:: 2 + skew]:
                name = content[idx].replace(">", "").replace("\n", "").strip()
                seq = content[idx + 1].replace("\n", "")
                res_dict[name] = seq
    return res_dict


def parse_pdb(filepath):
    structure = parser.get_structure(filepath, filepath)[0]
    protein = {}
    # chain level
    for j, chain in enumerate(structure.get_chains()):
        aa_resnames = []
        aa_residues_dict = {}
        # residue level
        for i, residue in enumerate(chain.get_residues()):
            res_id = residue.get_id()
            resname = residue.resname.strip()
            if res_id[0] == " ":
                aa_resnames.append(resname)
                aa_residues_dict[i] = {"resname": resname}
                # atom level
                atomnames = []
                residue_atomcoords = []
                for atom in residue.get_atoms():
                    if atom.element != "H":
                        atomname = atom.get_name()
                        atomnames.append(atomname)
                        residue_atomcoords.append(list(atom.get_vector()))
                aa_residues_dict[i]["atoms"] = atomnames
                aa_residues_dict[i]["coords"] = np.array(residue_atomcoords)
        # not save non-standard residues
        if len(aa_resnames) == 0:
            continue
        protein[j] = {"aa_residues": aa_residues_dict}
        protein[j]["chain_id"] = chain.id
        aa_seq = seq1("".join(aa_resnames))
        protein[j]["aa_seq"] = aa_seq
    return protein


def parse_sdf(file_path):
    suppl = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=True)
    molecules = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
    if len(molecules) == 0:
        return Chem.SDMolSupplier(file_path, sanitize=False, removeHs=False)  # if no valid molecules found, return unsanitized
    return molecules


def pp2dict(prot_file, pokt_file):
    protein = parser.get_structure(prot_file, prot_file)[0]  # type: ignore
    pocket = parser.get_structure(pokt_file, pokt_file)[0]  # type: ignore
    res_dict = {}
    index_dict = {}
    order_list = []
    for chain in protein:
        # fasta
        aa_resnames = [residue.resname.strip() for residue in chain if residue.get_id()[0] == " "]
        if len(aa_resnames) > 0:
            res_dict[chain.id] = seq1("".join(aa_resnames))
            order_list.append(chain.id)
            if chain.id in pocket:
                pocket_chain_set = set([res.get_id() for res in pocket[chain.id]])
                pkt_idx = [res.get_id() in pocket_chain_set for res in chain if res.id[0] == " "]
                # 将bool数组直接转换为01字符串
                index_dict[chain.id] = "".join("1" if x else "0" for x in pkt_idx)
    res_dict["order"] = order_list
    return {"aa_seq": res_dict, "p_idx": index_dict}

