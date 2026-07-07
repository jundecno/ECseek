import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import *


# 筛选指定分子
def get_all_smiles(pred_dir):
    smis = set()
    for pred_file in os.listdir(pred_dir):
        lines = read_lines(os.path.join(pred_dir, pred_file))
        for line in lines:
            rxn, score = line.strip().split(",")
            cons, prod = rxn.split(">>")
            smis.update(cons.split("."))
            smis.update(prod.split("."))
    res_str = "\n".join(smis)
    write_txt(os.path.join(pred_dir, "all_smiles.txt"), res_str)


def get_max_similarity(smiles_txt, target_txt, out_file):
    make_dir(os.path.dirname(out_file))
    smiles = read_lines(smiles_txt)
    targets = read_lines(target_txt)
    target_smis = [smi.split()[0] for smi in targets]
    target_names = [smi.split()[1] for smi in targets]
    # 计算相似度
    for smi in smiles:
        for target_smi, target_name in zip(target_smis, target_names):
            sim = get_similarity(smi, target_smi)
            if sim > 0.5:  # 设置相似度阈值
                append_txt(out_file, f"Smiles: {smi}, Target: {target_name}, Similarity: {sim:.4f}\n")  

if __name__ == "__main__":
    for pocket_dir in os.listdir("/data/zzjun/EnzySeek/data/steroid/special/prediction/"):
        # get_all_smiles(f"/data/zzjun/EnzySeek/data/steroid/special/prediction/{pocket_dir}")
        get_max_similarity(
            f"/data/zzjun/EnzySeek/data/steroid/special/prediction/{pocket_dir}/all_smiles.txt", "/data/zzjun/EnzySeek/data/steroid/target.smi",
            f"/data/zzjun/EnzySeek/data/steroid/special/results/{pocket_dir}_simis.txt",
        )
