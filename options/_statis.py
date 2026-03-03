import pandas as pd
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))

def cal_prot_counts():
    df = pd.read_csv(f"{root_path}/data/Enzyme/ENZYME/enzyme.csv")
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

if __name__ == "__main__":
    # cal_prot_counts()
    statis_prots(f"/data/zzjun/ECseek/data/Enzyme/ENZYME/af2_error_seqs.fa")
