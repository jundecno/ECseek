import rootutils
root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from utils import *
import requests
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from threading import local
import urllib3
# download alphafill data example: https://www.alphafill.eu/v1/aff/P02768 https://www.alphafill.eu/v1/aff/P02768

_thread_local = local()
afilldb_url = "http://www.alphafill.eu/v1/aff/"

def get_session():
    """Thread-local requests session"""
    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
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
def download_afilldb(pdb_id, ec_number):
    # 从afilldb下载所有结构
    save_dir = f"{root_path}/data/Enzyme/ENZYME/afilldb/{ec_number}"
    os.makedirs(save_dir, exist_ok=True)
    cif_path = os.path.join(save_dir, f"{pdb_id}.cif")
    if os.path.exists(cif_path): return
    
    session = get_session()
    url, path = f"{afilldb_url}{pdb_id}", cif_path
    print(url)
    
    try:
        r = session.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return
    except Exception as e:
        print(e)

    with open("afill_failed_pdbs.txt", "a") as f:
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
        list(tqdm(executor.map(lambda pair: download_afilldb(pair[0], pair[1]), ec_pairs), total=len(ec_pairs), desc="Downloading PDBs"))



if __name__=='__main__':
    # download_all(f"{root_path}/data/Enzyme/ENZYME/enzyme.csv")
    download_afilldb("P22811","1.17.1.4")