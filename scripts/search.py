import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import *
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.model import TaskModel, set_all_seed
from datetime import datetime

from models.dataloader import TrainDataset, DataLoader, TestDataset
import lightning.pytorch as pl
import torch.nn.functional as F

CKPT_PATH = "/data/zzjun/EnzySeek/checkpoints/try/2026-05-01_23-37-59/epoch_099.ckpt"

def search_rxn(pocket):
    torch.set_float32_matmul_precision("high")
    pred_dataset = TestDataset(f"{root_path}/data/features", [pocket], "poc")
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, follow_batch=["mol_cls"])
    model = TaskModel.load_from_checkpoint(CKPT_PATH, weights_only=False).cuda()
    model.eval()
    trainer = pl.Trainer(devices=[0], logger=False, accelerator="gpu")
    outs = trainer.predict(model, dataloaders=pred_dataloader)
    out = outs[0].cpu().numpy().reshape(-1)  # type: ignore
    rxn_dict = pkl_load(f"{root_path}/results/Enzyme-405_2026-05-01_23-37-59_epoch_099_rxn_embed.pkl")
    cos_sims = search_similarity(out, rxn_dict, top_k=10)
    for k, v in cos_sims:
        print(k, F.sigmoid(torch.tensor(v * 4.603962421417236)).numpy())  # type: ignore


def search_poc(rxn):
    torch.set_float32_matmul_precision("high")
    pred_dataset = TestDataset(f"{root_path}/data/features", [rxn], "rxn")
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, follow_batch=["mol_cls"])
    model = TaskModel.load_from_checkpoint(CKPT_PATH, weights_only=False).cuda()
    model.eval()
    trainer = pl.Trainer(devices=[0], logger=False, accelerator="gpu")
    outs = trainer.predict(model, dataloaders=pred_dataloader)
    out = outs[0].cpu().numpy().reshape(-1)  # type: ignore
    poc_dict = pkl_load(f"{root_path}/results/Enzyme-405_2026-05-01_23-37-59_epoch_099_poc_embed.pkl")
    cos_sims = search_similarity(out, poc_dict, top_k=10)
    for k, v in cos_sims:
        print(k, F.sigmoid(torch.tensor(v * 4.603962421417236)).numpy())
        
if __name__ == "__main__":
    search_rxn("A0A0D2H023")
    search_poc("*/C(S)=N/OS(=O)(=O)O>>*C#N.O=S(=O)(O)O.S")
