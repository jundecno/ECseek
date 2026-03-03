import rootutils
root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from threading import local
from Bio import SeqIO
import gemmi


_thread_local = local()
afdb_url = "https://alphafold.ebi.ac.uk/files/"

def is_cif_complete(filepath):
    try:
        doc = gemmi.cif.read_file(filepath)
        return True
    except Exception:
        return False

def is_pdb_complete(filepath):
    try:
        structure = gemmi.read_structure(filepath)
        return True
    except Exception:
        return False

def get_session():
    """Thread-local requests session"""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


# https://alphafold.ebi.ac.uk/files/AF-Q6AZW2-F1-model_v6.cif
def download_afdb(pdb_id, ec_number):
    # 从afdb下载所有结构
    save_dir = f"{root_path}/data/Enzyme/ENZYME/afdb/{ec_number}"
    os.makedirs(save_dir, exist_ok=True)
    pdb_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    cif_path = os.path.join(save_dir, f"{pdb_id}.cif")
    
    if is_cif_complete(cif_path):
        return
    
    session = get_session()
    urls = [
        (f"{afdb_url}AF-{pdb_id}-F1-model_v6.cif", cif_path),
        (f"{afdb_url}AF-{pdb_id}-F1-model_v6.pdb", pdb_path),
        (f"{afdb_url}AF-{pdb_id}-F1-model_v4.cif", cif_path),
        (f"{afdb_url}AF-{pdb_id}-F1-model_v4.pdb", pdb_path),
    ]
    for url, path in urls:
        tmp = path + ".tmp"
        r = session.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, path)  # 下载完成后重命名
            return

    # 如果两种格式都失败
    with open("af2_failed_pdbs.txt", "a") as f:
        f.write(pdb_id + "\n")


def download_all(csv_file):
    df = pd.read_csv(csv_file)
    uniprot_ids = df["UniProt_IDs"].values
    ec_numbers = df["EC_Number"].values
    ec_pairs = []
    for i, (uniprot_str, ec) in enumerate(zip(uniprot_ids, ec_numbers)):
        ec = ec.replace(".", "/")
        uniprot_list = uniprot_str.split(";")
        for uniprot in uniprot_list:
            ec_pairs.append((uniprot, ec))
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(lambda pair: download_afdb(pair[0], pair[1]), ec_pairs), total=len(ec_pairs), desc="Downloading PDBs"))


if __name__ == "__main__":
    # download_all(f"{root_path}/data/Enzyme/ENZYME/enzyme.csv")
    print(is_cif_complete("/data/zzjun/ECseek/data/Enzyme/ENZYME/afdb/5/4/2/10/Q3SJR2.cif"))
