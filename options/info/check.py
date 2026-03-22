# 检查是否存在可遇见的问题
import shutil
import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
from Bio.PDB import MMCIFParser  # type: ignore

def check_enzyme_seqs(csv_file, uid2seq_file):
    uid2seq = json_load(uid2seq_file)
    df = pd.read_csv(filepath_or_buffer=csv_file)
    uniprot_ids = df["UniProt_IDs"].values
    uids = set()
    for uniprot_str in uniprot_ids:
        uniprot_list = uniprot_str.split(";")
        uids.update(uniprot_list)
    uid2seq_set = set(uid2seq.keys())
    missing_uids = uids - uid2seq_set

    if missing_uids:
        print(f"Missing UniProt IDs: {missing_uids}")
    else:
        print("All UniProt IDs have corresponding sequences in the uid2seq mapping.")


def check_rhea_seqs(csv_file, uid2seq_file):
    uid2seq = json_load(uid2seq_file)
    df = pd.read_csv(csv_file, sep="\t")
    uniprot_ids = df["ID"].values
    uids = set(uniprot_ids)
    uid2seq_set = set(uid2seq.keys())
    missing_uids = uids - uid2seq_set

    if missing_uids:
        print(f"Missing UniProt IDs: {missing_uids}")
    else:
        print("All UniProt IDs have corresponding sequences in the uid2seq mapping.")


def get_af2_error(json_file, save_file):
    uid2seq = json_load(json_file)
    uid2seq_set = set(uid2seq.keys())
    af2_dir = f"{root_path}/data/afdb/"
    af2_files = []
    for root, dirs, files in os.walk(af2_dir):
        for file in files:
            af2_files.append(file)
    af2_uids = set([file.split(".")[0] for file in af2_files])
    missing_uids = uid2seq_set - af2_uids
    # write error fasta
    with open(save_file, "w") as f:
        for uid in missing_uids:
            seq = uid2seq[uid]
            f.write(f">{uid}\n{seq}\n")
    print(f"Missing UniProt IDs in AF2 database: {len(missing_uids)}")


def is_cif(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "loop_" in line:
                return True
    return False


def is_afdb(file_path):
    with open(file_path, "r") as f:
        head_line = f.readline()
        if "data_AF" in head_line:
            return True
    return False


def check_no_af2(afill_dir):
    uid_list = []
    all_files = tranverse_folder(afill_dir)

    for filename in all_files:
        if ".cif" in filename:
            uid_list.append(filename.split(".")[0])
    fa_dict = json_load(f"{root_path}/data/enzyme/RHEA/uid2seq.json")
    for uid in tqdm(uid_list):
        file_id = os.path.basename(uid)
        file_path = f"{uid}.cif"
        if not is_cif(file_path):
            append_txt(f"{root_path}/data/enzyme/RHEA/af2_err.fa", f">{file_id}\n{fa_dict[file_id]}\n")


def check_multi_chain(afdb_dir):
    all_files = tranverse_folder(afdb_dir)  # 假设你已定义此函数
    multi_chain_files = []

    for filename in tqdm(all_files):
        if filename.endswith(".cif") or filename.endswith(".cif.gz"):
            try:
                # gemmi.read_structure 自动处理压缩和非压缩文件
                doc = gemmi.cif.read_file(filename)
                st = gemmi.make_structure_from_block(doc[0])
                # 获取第一模型的链列表
                # Gemmi 的 st[0] 访问第一个 Model
                chains = [chain.name for chain in st[0]]
                if len(chains) > 1:
                    multi_chain_files.append(filename)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    print(f"Files with multiple chains: {len(multi_chain_files)}") # 0
    for file in multi_chain_files:
        print(file)


def check_train_valid(csv_file):
    # 检查csv的UniProtID列是存在对应cif文件
    df = pd.read_csv(csv_file)
    uniprot_ids = df["UniprotID"].values
    missing_uids = []
    for uniprot_id in tqdm(uniprot_ids):
        cif = f"{AFDB}/{uid2path(uniprot_id)}"
        # print(cif)
        if not os.path.exists(cif):
            missing_uids.append(uniprot_id)
    if missing_uids:
        print(f"Missing UniProt IDs in AFDB: {set(missing_uids)}")
    else:
        print("All UniProt IDs in the csv file have corresponding CIF files in AFDB.")


def check_pocket(sites_file):
    data_dict = pkl_load(sites_file)
    uid_dict = {}
    cnt = 0
    for uid, site_info in data_dict.items():
        residues = site_info["residues"]
        comp_id = site_info["comp_id"]
        if comp_id is not None:
            print(comp_id)
        elif residues is not None:    
            print(residues)
        elif comp_id is None and residues is None:
            cnt += 1
    print(cnt)


def check_rxn(train_csv, valid_csv, rxn2rc):
    rxn_rc_dict = pkl_load(rxn2rc)
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    all_df = set(train_df[RXN_COL].unique()) | set(valid_df[RXN_COL].unique())
    missing_rxns = set()
    for rxn in tqdm(all_df):
        if rxn not in rxn_rc_dict:
            missing_rxns.add(rxn)
    print(f"Missing reactions in rxn2rc mapping: {len(missing_rxns)}")


def is_valid_pocket_file(file_path):
    mmcif_parser = MMCIFParser(QUIET=True)
    try:
        structure = mmcif_parser.get_structure("pocket", file_path)
        return True
    except Exception as e:
        return False


def check_pocket_file(dir):
    files = get_file_paths(dir)
    for file in tqdm(files):
        if not is_valid_pocket_file(file):
            append_txt("err_poc.txt",f"{file}\n")


def check_prot_fa(uid2seq_file):
    uid2seq = pkl_load(uid2seq_file)
    tasks = []
    fa_len = {}
    for uid in tqdm(uid2seq):
        seq = uid2seq[uid]
        fasta_len = len(seq)
        prot_file_path = f"{AFDB}/{uid2path(uid)}"
        fa_len[uid] = fasta_len
        tasks.append(prot_file_path)

    pool = mlc.SuperPool(128)
    len_list = pool.map(get_resi_len, tasks,32)

    cnt = 0
    mismatch_list = []
    for uid_len in len_list:
        uid, stru_len = uid_len
        fasta_len = fa_len[uid]

        if fasta_len != stru_len:
            cnt += 1
            mismatch_list.append((uid, fasta_len, stru_len))

    for uid, fasta_len, prot_len in mismatch_list:
        print(f"UID: {uid}, FASTA Length: {fasta_len}, Protein Length: {prot_len}")
    print(f"Total mismatches: {cnt}")
    # UID: P49328, FASTA Length: 19, Protein Length: 532

def check_final_sites(sites_file):
    data_dict = pkl_load(sites_file)
    print(len(data_dict))
    for uid, site_info in data_dict.items():
        residues = site_info["residues"]
        comp_id = site_info["comp_id"]
        # if comp_id is not None:
        #     print(comp_id)
        # elif residues is not None:    
        #     print(residues)
        if comp_id is None and residues is None:
            print(uid)

if __name__ == "__main__":
    # check_enzyme_seqs(f"{root_path}/data/enzyme/ENZYME/enzyme.csv", f"{root_path}/data/enzyme/ENZYME/uid2seq.json")
    # check_rhea_seqs(f"{root_path}/data/enzyme/RHEA/rhea2uniprot_sprot.tsv", f"{root_path}/data/enzyme/RHEA/uid2seq.json")
    # json2fasta(f"{root_path}/data/enzyme/ENZYME/uid2seq.json", f"{root_path}/data/enzyme/ENZYME/uid2seq.fasta")
    # json2fasta(f"{root_path}/data/enzyme/RHEA/uid2seq.json", f"{root_path}/data/enzyme/RHEA/uid2seq.fasta")
    # get_af2_error(f"{root_path}/data/enzyme/ENZYME/uid2seq.json", f"{root_path}/data/enzyme/ENZYME/af2_error.fasta")
    # get_af2_error(f"{root_path}/data/enzyme/RHEA/uid2seq.json", f"{root_path}/data/enzyme/RHEA/af2_error.fasta")

    # find /data/zzjun/ECseek/data/afdb -name "*.cif" -type f | wc -l
    # find /data/zzjun/ECseek/data/afilldb -name "*.json" -type f | wc -l
    # check_no_cf2("/data/zzjun/ECseek/data/afdb")
    # check_no_af2(f"{root_path}/data/afilldb/")Q5TJ03
    # check_multi_chain(AFDB)
    # check_no_af2(AFDB)
    # A5EW83
    # fa_dict = json_load(f"{root_path}/data/enzyme/RHEA/uid2seq.json")
    # print(fa_dict["A5EW83"])
    # check_train_valid(f"{root_path}/data/enzyme/RHEA/split/train.csv")
    # check_train_valid(f"{root_path}/data/enzyme/RHEA/split/valid.csv")
    # check_pocket(f"{root_path}/data/enzyme/RHEA/proc/final_sites.pkl")
    # data = pkl_load("/data/zzjun/ECseek/data/features/center/rxn2rc_localmapper.pkl")
    # print(len(data))
    # check_rxn(f"{root_path}/data/training/train.csv", f"{root_path}/data/training/valid.csv", f"{root_path}/data/features/center/rxn2rc_rxnmapper.pkl")
    # check_pocket_file(f"{root_path}/data/pockets/")
    # check_prot_fa("/data/zzjun/ECseek/data/enzyme/RHEA/uid2seq.pkl")
    # check_final_sites(f"{root_path}/data/enzyme/RHEA/proc/final_sites.pkl")
