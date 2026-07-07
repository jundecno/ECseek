# Check script

# /data/zzjun/EnzySeek/results/Enzyme-405_ec_nvn_2026-05-01_23-37-59_epoch_099_rxn_embed.pkl
# /data/zzjun/EnzySeek/results/Enzyme-405_ec_nvn_2026-06-01_09-10-32_epoch_099_rxn_embed.pkl

# 检查两个嵌入字典相似是否够多

import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
import rootutils
from models.dataloader import TrainDataset, DataLoader, TestDataset

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import *
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from models.model import TaskModel

def check_embed_similarity(poc_embed_file1, poc_embed_file2, rxn_embed_file1, rxn_embed_file2):
    poc_embed_dict1 = pkl_load(poc_embed_file1)
    poc_embed_dict2 = pkl_load(poc_embed_file2)
    rxn_embed_dict1 = pkl_load(rxn_embed_file1)
    rxn_embed_dict2 = pkl_load(rxn_embed_file2)

    # 计算相似度
    uids = list(poc_embed_dict1.keys())
    rxns = list(rxn_embed_dict1.keys())
    poc_feats1 = np.array([poc_embed_dict1[uid] for uid in uids])
    poc_feats2 = np.array([poc_embed_dict2[uid] for uid in uids])
    rxn_feats1 = np.array([rxn_embed_dict1[rxn] for rxn in rxns])
    rxn_feats2 = np.array([rxn_embed_dict2[rxn] for rxn in rxns])

    poc_sim = np.mean([F.cosine_similarity(torch.tensor(poc_feats1[i]), torch.tensor(poc_feats2[i]), dim=0).item() for i in range(len(uids))])
    rxn_sim = np.mean([F.cosine_similarity(torch.tensor(rxn_feats1[i]), torch.tensor(rxn_feats2[i]), dim=0).item() for i in range(len(rxns))])

    print(f"Average cosine similarity for poc embeddings: {poc_sim:.4f}")
    print(f"Average cosine similarity for rxn embeddings: {rxn_sim:.4f}")


def check_rxn_similarity(rxn_embed_file1, rxn_embed_file2):
    rxn_embed_dict1 = pkl_load(rxn_embed_file1)
    rxn_embed_dict2 = pkl_load(rxn_embed_file2)
    # 计算相似度
    rxns = list(rxn_embed_dict1.keys())
    rxn_feats1 = np.array([rxn_embed_dict1[rxn] for rxn in rxns])
    rxn_feats2 = np.array([rxn_embed_dict2[rxn] for rxn in rxns])
    rxn_sim = np.mean([F.cosine_similarity(torch.tensor(rxn_feats1[i]), torch.tensor(rxn_feats2[i]), dim=0).item() for i in range(len(rxns))])
    print(f"Average cosine similarity for rxn embeddings: {rxn_sim:.4f}")


def check_lmdb_index_error(lmdb_file):
    env = lmdb.open(lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    bad_samples = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            # 是否存在nan
            if np.isnan(np.array(pkl.loads(value).x)).any():
                bad_samples.append((key, "Contains NaN in node features"))
                print(f"Bad sample key: {key}, error: Contains NaN in node features")
                continue
            # try:
            #     data = pkl.loads(value)

            #     # 检查节点特征
            #     if not hasattr(data, "x") or data.x is None:
            #         raise ValueError("Missing x attribute")
            #     num_nodes = data.x.size(0)

            #     # 检查 edge_index
            #     if hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            #         edge_index = data.edge_index
            #         if edge_index.dtype != torch.long:
            #             raise ValueError(f"edge_index dtype {edge_index.dtype} invalid")
            #         if edge_index.size(0) != 2:
            #             raise ValueError(f"edge_index shape {tuple(edge_index.shape)} invalid")
            #         if edge_index.min() < 0:
            #             raise ValueError(f"edge_index contains negative index: {int(edge_index.min())}")
            #         if edge_index.max() >= num_nodes:
            #             raise ValueError(f"edge_index out of range: max={int(edge_index.max())}, num_nodes={num_nodes}")

            #     # 检查 edge_attr
            #     if hasattr(data, "edge_attr") and data.edge_index.numel() > 0:
            #         if data.edge_attr.size(0) != data.edge_index.size(1):
            #             raise ValueError(f"edge_attr rows {data.edge_attr.size(0)} != num_edges {data.edge_index.size(1)}")

            # except Exception as e:
            #     bad_samples.append((key, repr(e)))
            #     print(f"Bad sample key: {key}, error: {e}")

    env.close()
    print(f"Total bad samples: {len(bad_samples)}")
    return bad_samples


def create_rxn_embed(rxns, save_file, ckpt_path, is_save=True):
    torch.set_float32_matmul_precision("high")
    rxn_list = list(rxns)
    pred_dataset = TestDataset(f"{root_path}/data/features", rxn_list, "rxn")
    pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=1, follow_batch=["mol_cls"])
    model = TaskModel.load_from_checkpoint(ckpt_path, weights_only=False).cuda()
    model.eval()
    trainer = pl.Trainer(devices=[1], logger=False, accelerator="gpu")
    outs = trainer.predict(model, dataloaders=pred_dataloader)
    outs = torch.cat(outs, dim=0)  # type: ignore [b, emb_dim]
    res_dict = {}
    for rxn, emb in zip(rxn_list, outs):  # type: ignore
        res_dict[rxn] = emb.detach().to(dtype=torch.float32, device="cpu").numpy().reshape(-1)
    if is_save:
        pkl_dump(save_file, res_dict)
    return res_dict


if __name__ == "__main__":
    BATCH_SIZE = 16
    # Enzyme-405_ec_2026-06-23_19-51-12_epoch_094_poc_embed
    # poc_embed_file1 = f"{root_path}/results/Enzyme-405_ec_2026-06-23_19-51-12_epoch_094_poc_embed.pkl"
    # poc_embed_file2 = f"{root_path}/results/Enzyme-405_ec_2026-06-23_19-51-12_epoch_094_poc_embed1.pkl"
    rxn_embed_file1 = f"{root_path}/results/Enzyme-405_ec_2026-06-23_19-51-12_epoch_094_rxn_embed.pkl"
    rxn_embed_file2 = f"{root_path}/results/Enzyme-405_ec_2026-06-23_19-51-12_epoch_094_rxn_embed1.pkl"
    df = pd.read_csv(f"{root_path}/results/Enzyme-405_ec.csv")
    rxns = list(df["CANO_RXN_SMILES"].unique())
    create_rxn_embed(rxns, rxn_embed_file1, "/data/zzjun/EnzySeek/checkpoints/new/2026-06-23_19-51-12/epoch_094.ckpt")
    check_rxn_similarity(rxn_embed_file1, rxn_embed_file2)
    # check_lmdb_index_error(f"{root_path}/data/features/poc_graph.lmdb")
    # check_lmdb_index_error(f"{root_path}/data/features/rxn_graph.lmdb")
