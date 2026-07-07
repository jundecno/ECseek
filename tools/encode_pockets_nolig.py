#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os
import sys
import pickle

import os
from Bio.PDB import PDBParser
import numpy as np
import lmdb
import numpy as np
import pickle


def write_lmdb(data, lmdb_path, num):
    # resume

    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode("ascii"), pickle.dumps(d))
            num += 1

    return num


def pocket2lmdb(pocket_name, biopy_chain, pdb_name):
    recpt = list(biopy_chain.get_atoms())
    pocket_atom_type = [x.element for x in recpt if x.element != "H"]
    pocket_coord = [x.coord for x in recpt if x.element != "H"]
    print(pdb_name + "_" + pocket_name, len(pocket_atom_type))
    return {"pocket": pdb_name + "_" + pocket_name, "pocket_atoms": pocket_atom_type, "pocket_coordinates": pocket_coord}


def process_one_pdbdir(dirs, name="pocket"):
    all_pocket = []
    for d in os.listdir(dirs):
        try:
            p = PDBParser()
            model = p.get_structure("0", os.path.join(dirs, d))[0] # type: ignore
            pdb_name = d.split(".")[0]
            pocket = [pocket2lmdb(pdb_name.replace("pocket", ""), model, pdb_name)]
            all_pocket += pocket
        except:
            pass
    if os.path.exists(os.path.join(dirs, f"{name}.lmdb")):
        return 0
    write_lmdb(all_pocket, os.path.join(dirs, f"{name}.lmdb"), 0)
    return 1

def cli_main():
    # add args
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket-dir","-p", type=str, default="", help="path for pocket dir")
    args = parser.parse_args()

    for pocket_dir in os.listdir(args.pocket_dir):
        if not os.path.exists(os.path.join(args.pocket_dir, pocket_dir, "pocket.lmdb")):
            ret = process_one_pdbdir(os.path.join(args.pocket_dir, pocket_dir))


if __name__ == "__main__":
    cli_main()
