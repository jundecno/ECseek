import rootutils
root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import csv
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from threading import local
from Bio import SeqIO

def parse_enzyme_dat(dat_file, csv_file):
    headers = ["EC_Number", "Official_Name", "Alternative_Names", "Reaction", "UniProt_IDs", "UniProt_Names"]
    rows = []
    current_entry = {"ID": "", "DE": [], "AN": [], "CA": [], "DR": [], "NA": []}

    with open(dat_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tag = line[:2]
            content = line[2:].strip()
            if tag == "ID":
                current_entry["ID"] = content
            elif tag == "DE":
                if "Transferred entry" in content or "Deleted entry" in content:
                    current_entry["ID"] = ""
                else:
                    current_entry["DE"].append(content.rstrip("."))
            elif tag == "AN":
                current_entry["AN"].append(content.rstrip("."))
            elif tag == "CA":
                current_entry["CA"].append(content.rstrip("."))
            elif tag == "DR":
                # 成对捕获：Accession (Q9X6U2) 和 Entry Name (BDHA_CUPNH)
                pairs = re.findall(r"([A-Z0-9]+),\s+([A-Z0-9_]+)\s*;", content)
                for acc, name in pairs:
                    current_entry["DR"].append(acc)
                    current_entry["NA"].append(name)
            elif tag == "//":
                if current_entry["ID"] and len(current_entry["DR"]) != 0:
                    row = {
                        "EC_Number": current_entry["ID"],
                        "Official_Name": " ".join(current_entry["DE"]),
                        "Alternative_Names": " | ".join(current_entry["AN"]),
                        "Reaction": " ".join(current_entry["CA"]),
                        "UniProt_IDs": ";".join(current_entry["DR"]),
                        "UniProt_Names": ";".join(current_entry["NA"]),
                    }
                    rows.append(row)
                current_entry = {"ID": "", "DE": [], "AN": [], "CA": [], "DR": [], "NA": []}

    with open(csv_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def get_fasta(csv_file, uniprot_file):
    df = pd.read_csv(csv_file)
    uniprot_ids = df["UniProt_IDs"].values
    ec_numbers = df["EC_Number"].values
    ec_pairs = []
    for i, (uniprot_str, ec) in enumerate(zip(uniprot_ids, ec_numbers)):
        uniprot_list = uniprot_str.split(";")
        for uniprot in uniprot_list:
            ec_pairs.append((uniprot, ec))

    records = SeqIO.parse(uniprot_file, "fasta")
    seq_set = {}
    for record in records:
        uniprot_id = record.id.split("|")[1]  # UniProt ID通常在第二部分
        seq = str(record.seq)
        seq_set[uniprot_id] = seq

    res_str = ""
    print(f"Total EC pairs: {len(ec_pairs)}")
    for ec_pair in ec_pairs:
        uniprot_id, ec = ec_pair
        if uniprot_id in seq_set:
            res_str += f">{uniprot_id} EC:{ec}\n"
            res_str += seq_set[uniprot_id] + "\n"
        else:
            print(f"Warning: UniProt ID {uniprot_id} not found in FASTA file.")
    write_txt(f"{root_path}/data/Enzyme/ENZYME/uniprot_sprot_ec.fasta", res_str)

if __name__ == "__main__":
    # parse_enzyme_dat("/home/saber/projs/ECseek/data/Enzyme/ENZYME/enzyme.dat", "/home/saber/projs/ECseek/data/Enzyme/ENZYME/enzyme.csv")
    get_fasta(f"{root_path}/data/Enzyme/ENZYME/enzyme.csv", f"{root_path}/data/Enzyme/ENZYME/uniprot_sprot.fasta")
    pass