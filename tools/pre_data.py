import os
from Bio.PDB import PDBParser, PDBIO
import numpy
import rootutils


root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *

def add_chain_a_to_pdb(input_dir, output_dir):
    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pdb"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                # 解析 PDB 结构
                structure = parser.get_structure(file_name, input_path)

                modified = False
                for model in structure:
                    for chain in model:
                        # 检查 Chain ID 是否为空、空格或标识为 ' '
                        if not chain.id or chain.id == " " or chain.id == "":
                            chain.id = "A"
                            modified = True

                # 保存修改后的文件
                io.set_structure(structure)
                io.save(output_path)

                status = "Fixed -> Chain A" if modified else "Kept original"
                print(f"Processed {file_name}: {status}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # 使用示例
    # input_folder = "/data01/zzjun/ECseek/help_data/Rg_pocket/all_struc"  # 你的原始 PDB 文件夹
    # add_chain_a_to_pdb(input_folder, input_folder)  # 输出到同一文件夹，覆盖原文件
    # from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    # res_dict = {}
    # generator = GetMorganGenerator(2, fpSize=2048)
    # data = pkl_load("/data01/zzjun/ECseek/tools/Drug-The-Whole-Genome/data/encoded_mol_embs/6_folds/fold0.pkl")
    # smiles = data[1]
    # for smi in tqdm(smiles):
    #     hit, smi = smi.split(",")
    #     mol = Chem.MolFromSmiles(smi)
    #     if not mol is None:
    #         res_dict[hit] = generator.GetFingerprint(mol)
    # pkl_dump("/data01/zzjun/ECseek/help_data/hit.pkl",res_dict)

    # data = json_load("/data01/zzjun/ECseek/help_data/steroid.json")
    # for name in tqdm(data):
    #     smi = data[name]
    #     mol = Chem.MolFromSmiles(smi)
    #     res_dict[name] = generator.GetFingerprint(mol)
    # pkl_dump("/data01/zzjun/ECseek/help_data/steroid.pkl", res_dict)

    data = pkl_load("/data01/zzjun/ECseek/help_data/Rg_results/hit.pkl")
    df = pd.DataFrame(columns=["pdb_id", "ligand", "max_sim"])
    for key in data:
        if len(data[key]["ligand"]) > 0:
            df = pd.concat([df, pd.DataFrame({"pdb_id": key, "ligand": [data[key]["ligand"]], "max_sim": [data[key]["max_sim"]]})], ignore_index=True)
    df.to_csv("/data01/zzjun/ECseek/help_data/Rg_results/hit.csv", index=False)
