import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import *
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.model import TaskModel
from datetime import datetime
import torch.nn.functional as F
from models.dataloader import TrainDataset, DataLoader, TestDataset
import lightning.pytorch as pl
from models.model import calc_diagonal_sim

RESULTS_PATH = f"{root_path}/results/"


def create_poc_embed(uids, save_file, ckpt_path, is_save=True):
    torch.set_float32_matmul_precision("high")
    poc_list = list(uids)
    pred_dataset = TestDataset(f"{root_path}/data/features", poc_list, "poc")
    pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=1, follow_batch=["mol_cls"])
    model = TaskModel.load_from_checkpoint(ckpt_path, weights_only=False).cuda()
    model.eval()
    trainer = pl.Trainer(devices=[1], logger=False, accelerator="gpu")
    outs = trainer.predict(model, dataloaders=pred_dataloader)  # [emb1, emb2, ...]
    outs = torch.cat(outs, dim=0)  # type: ignore [b, emb_dim]
    res_dict = {}
    for poc, emb in zip(poc_list, outs):  # type: ignore
        res_dict[poc] = emb.detach().to(dtype=torch.float32, device="cpu").numpy().reshape(-1)
    if is_save:
        pkl_dump(save_file, res_dict)
    return res_dict  # , model.loss_fn.logit_scale.cpu().item()  # type: ignore


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


def single_eval(test_data_type, ckpt_path):
    name = f"{test_data_type}_{dir2name(ckpt_path)}"
    csv_file = f"{RESULTS_PATH}/{test_data_type}.csv"

    if not os.path.exists(f"{RESULTS_PATH}/{name}_results.csv"):
        df = pd.read_csv(csv_file)
        uids = list(df["UniprotID"].unique())
        rxns = list(df["CANO_RXN_SMILES"].unique())
        poc_embed_dict = (
            create_poc_embed(uids, f"{RESULTS_PATH}/{name}_poc_embed.pkl", ckpt_path)
            # if not os.path.exists(f"{RESULTS_PATH}/{name}_poc_embed.pkl")
            # else pkl_load(f"{RESULTS_PATH}/{name}_poc_embed.pkl")
        )
        rxn_embed_dict = (
            create_rxn_embed(rxns, f"{RESULTS_PATH}/{name}_rxn_embed.pkl", ckpt_path)
            # if not os.path.exists(f"{RESULTS_PATH}/{name}_rxn_embed.pkl")
            # else pkl_load(f"{RESULTS_PATH}/{name}_rxn_embed.pkl")
        )

        # 创建pred列，计算cosine similarity
        poc_feats = np.stack([poc_embed_dict[uid] for uid in uids])
        rxn_feats = np.stack([rxn_embed_dict[rxn] for rxn in rxns])

        poc_norms = np.linalg.norm(poc_feats, axis=1, keepdims=True)
        rxn_norms = np.linalg.norm(rxn_feats, axis=1, keepdims=True)

        poc_feats = poc_feats / np.maximum(poc_norms, 1e-8)
        rxn_feats = rxn_feats / np.maximum(rxn_norms, 1e-8)

        sims = poc_feats @ rxn_feats.T
        # 使用$$Score = \sigma(s \times \text{trained\_scale})$$
        # sims = F.sigmoid(torch.tensor(sims * logit_scale)).numpy()  # type: ignore
        sim_dict = {}
        for i, uid in enumerate(uids):
            sim_dict[uid] = {}
            for j, rxn in enumerate(rxns):
                sim_dict[uid][rxn] = sims[i][j]

        df["pred"] = df.apply(lambda x: sim_dict[x["UniprotID"]][x["CANO_RXN_SMILES"]], axis=1)
        df.to_csv(f"{RESULTS_PATH}/{name}_results.csv", index=False)
    else:
        df = pd.read_csv(f"{RESULTS_PATH}/{name}_results.csv")

    res_dict = evaluate_result(f"{RESULTS_PATH}/{name}_results.csv")
    print("\n########### Evaluation Results ###########")
    print(f'Top-1  SR : {res_dict["top1_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-3  SR : {res_dict["top3_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-5  SR : {res_dict["top5_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-10 SR : {res_dict["top10_sr"]*100:.3f}%')  # type: ignore
    print(f"Top-1% EF : {res_dict["top1_percent_ef"]:.4f}")
    print(f"Top-2% EF : {res_dict["top2_percent_ef"]:.4f}")
    print(f"Top-10 DCG: {res_dict["top10_dcg"]:.4f}")
    print("###########################################\n")
    if "ec" in name:
        print(
            f"{name}\t{res_dict["top1_sr"]*100:.3f}\t{res_dict["top3_sr"]*100:.3f}\t{res_dict["top5_sr"]*100:.3f}\t{res_dict["top10_sr"]*100:.3f}\t{res_dict["top10_dcg"]:.4f}"
        )
    else:
        print(
            f"{name}\t{res_dict["top1_sr"]*100:.3f}\t{res_dict["top3_sr"]*100:.3f}\t{res_dict["top5_sr"]*100:.3f}\t{res_dict["top10_sr"]*100:.3f}\t{res_dict["top1_percent_ef"]:.4f}\t{res_dict["top2_percent_ef"]:.4f}"
        )


def fold_eval(test_data_type, ckpt_path):
    name = f"fold_{test_data_type}_{dir2name(ckpt_path)}"
    csv_file = f"{RESULTS_PATH}/{test_data_type}.csv"

    if not os.path.exists(f"{RESULTS_PATH}/{name}_results.csv"):
        df = pd.read_csv(csv_file)
        uids = list(df["UniprotID"].unique())
        rxns = list(df["CANO_RXN_SMILES"].unique())
        ensemble_sims = np.zeros((len(uids), len(rxns)), dtype=np.float32)
        for fold in os.listdir(ckpt_path):
            fold_path = os.path.join(ckpt_path, fold)
            poc_embed_dict = create_poc_embed(uids, f"{RESULTS_PATH}/{name}_poc_embed.pkl", fold_path, False)
            rxn_embed_dict = create_rxn_embed(rxns, f"{RESULTS_PATH}/{name}_rxn_embed.pkl", fold_path, False)
            # 创建pred列，计算cosine similarity
            poc_feats = np.stack([poc_embed_dict[uid] for uid in uids])
            rxn_feats = np.stack([rxn_embed_dict[rxn] for rxn in rxns])

            poc_norms = np.linalg.norm(poc_feats, axis=1, keepdims=True)
            rxn_norms = np.linalg.norm(rxn_feats, axis=1, keepdims=True)

            poc_feats = poc_feats / np.maximum(poc_norms, 1e-8)
            rxn_feats = rxn_feats / np.maximum(rxn_norms, 1e-8)

            sims = poc_feats @ rxn_feats.T
            ensemble_sims += sims

        ensemble_sims /= len(os.listdir(ckpt_path))
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        rxn_to_idx = {rxn: j for j, rxn in enumerate(rxns)}

        df["pred"] = [ensemble_sims[uid_to_idx[uid], rxn_to_idx[rxn]] for uid, rxn in zip(df["UniprotID"], df["CANO_RXN_SMILES"])]
        df.to_csv(f"{RESULTS_PATH}/{name}_results.csv", index=False)
    else:
        df = pd.read_csv(f"{RESULTS_PATH}/{name}_results.csv")

    res_dict = evaluate_result(f"{RESULTS_PATH}/{name}_results.csv")
    print("\n########### Evaluation Results ###########")
    print(f'Top-1  SR : {res_dict["top1_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-3  SR : {res_dict["top3_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-5  SR : {res_dict["top5_sr"]*100:.3f}%')  # type: ignore
    print(f'Top-10 SR : {res_dict["top10_sr"]*100:.3f}%')  # type: ignore
    print(f"Top-1% EF : {res_dict["top1_percent_ef"]:.4f}")
    print(f"Top-2% EF : {res_dict["top2_percent_ef"]:.4f}")
    print(f"Top-10 DCG: {res_dict["top10_dcg"]:.4f}")
    print("###########################################\n")
    if "ec" in name:
        print(
            f"{name}\t{res_dict["top1_sr"]*100:.3f}\t{res_dict["top3_sr"]*100:.3f}\t{res_dict["top5_sr"]*100:.3f}\t{res_dict["top10_sr"]*100:.3f}\t{res_dict["top10_dcg"]:.4f}"
        )
    else:
        print(
            f"{name}\t{res_dict["top1_sr"]*100:.3f}\t{res_dict["top3_sr"]*100:.3f}\t{res_dict["top5_sr"]*100:.3f}\t{res_dict["top10_sr"]*100:.3f}\t{res_dict["top1_percent_ef"]:.4f}\t{res_dict["top2_percent_ef"]:.4f}"
        )


def eval_diag_sim(name, csv_file):
    # calc diag sim
    poc_embed_dict = pkl_load(f"{RESULTS_PATH}/{name}_poc_embed.pkl")
    rxn_embed_dict = pkl_load(f"{RESULTS_PATH}/{name}_rxn_embed.pkl")

    df = pd.read_csv(csv_file)
    df = df[df["Label"] == 1]
    uids = list(df["UniprotID"])
    rxns = list(df["CANO_RXN_SMILES"])
    uid_embs = np.stack([poc_embed_dict[uid] for uid in uids])
    rxn_embs = np.stack([rxn_embed_dict[rxn] for rxn in rxns])
    diag_sims = calc_diagonal_sim(torch.tensor(uid_embs), torch.tensor(rxn_embs)).numpy()
    print(f"Diagonal similarity: {diag_sims.mean():.4f}")


if __name__ == "__main__":
    test_data_type = "Enzyme-405"
    BATCH_SIZE = 16
    single_eval(test_data_type, "/data/zzjun/EnzySeek/checkpoints/new/2026-07-06_20-29-59/epoch_096.ckpt")
    # os.remove(f"{RESULTS_PATH}/{test_data_type}_2026-06-23_19-51-12_epoch_094_poc_embed.pkl")
    # os.remove(f"{RESULTS_PATH}/{test_data_type}_2026-06-23_19-51-12_epoch_094_rxn_embed.pkl")
    # eval_diag_sim("Enzyme-405_ec_2026-07-06_09-56-43_epoch_096", f"{root_path}/results/Enzyme-405_ec.csv")
    # fold_eval(test_data_type, "/data/zzjun/EnzySeek/checkpoints/new/2026-06-30_15-00-05")

    # single_eval("Enzyme-405_ec_2026-04-27_02-19-35_epoch_092", f"{root_path}/results/Enzyme-405_ec.csv", top_k=10)
    # create_rxn_embed(
    #     json_load(f"{root_path}/data/features/rxn2idx.json").keys(),
    #     f"/data/zzjun/EnzySeek/data/features/rxn_embed.pkl",
    # )
