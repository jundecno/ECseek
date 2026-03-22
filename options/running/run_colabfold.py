# colabfold_batch --templates --model-type alphafold2_ptm --num-recycle 3 --amber --num-models 5  /data/zzjun/ECseek/data/enzyme/ENZYME/af2_error.fasta /data/zzjun/ECseek/data/enzyme/ENZYME/colabfold --use-gpu-relax

# colabfold_batch --templates --model-type alphafold2_ptm --num-recycle 3 --amber --num-models 5  /data/zzjun/ECseek/options/info/tmp.fa /data/zzjun/ECseek/data/enzyme/RHEA/ofold --use-gpu-relax
# colabfold_batch --templates --model-type alphafold2_ptm --num-recycle 3 --amber --num-models 5  /data/zzjun/ECseek/data/enzyme/RHEA/af2_error.fasta /data/zzjun/ECseek/data/enzyme/ENZYME/colabfold --use-gpu-relax


# colabfold_batch --model-type alphafold2_ptm --num-recycle 3 --amber --num-models 1  test.fa ./tmp
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *

def get_cf2(file_dir, save_dir):
    # *_relaxed_rank_001*
    for file in os.listdir(file_dir):
        if "_relaxed_rank_001" in file:
            uid = file.split("_")[0]
            if os.path.exists(f"{save_dir}/{uid2path(uid)}"):
                continue
            else:
                save_file_path = f"{save_dir}/{uid2path(uid)}"
                make_dir(os.path.dirname(save_file_path))
                os.system(f"cp {file_dir}/{file} {save_file_path}")

def remove_relaxed(file_dir):
    for file in os.listdir(file_dir):
        if "_relaxed_rank" in file:
            uid = file.split("_")[0]
            # 删除所有uid开头的文件
            os.system(f"rm -r {file_dir}/{uid}*")

if __name__ == "__main__":
    get_cf2(f"{root_path}/data/enzyme/RHEA/ofold/", f"{root_path}/data/afdb/")
    # remove_relaxed(f"{root_path}/data/enzyme/ENZYME/colabfold/")
    pass
