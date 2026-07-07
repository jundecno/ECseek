import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import *
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.model import TaskModel, set_all_seed
from datetime import datetime

from models.dataloader import DataLoader, PocDataset
import lightning.pytorch as pl
import torch.nn.functional as F

CKPT_PATH = "/data/zzjun/EnzySeek/checkpoints/try/2026-05-01_23-37-59/epoch_099.ckpt"


def search_rxn(lmdb_file, out_file):
    make_dir(os.path.dirname(out_file))
    torch.set_float32_matmul_precision("high")
    pred_dataset = PocDataset(lmdb_file, "poc")
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, follow_batch=["mol_cls"])
    model = TaskModel.load_from_checkpoint(CKPT_PATH, weights_only=False).cuda()
    model.eval()
    trainer = pl.Trainer(devices=[0], logger=False, accelerator="gpu")
    outs = trainer.predict(model, dataloaders=pred_dataloader)
    rxn_dict = pkl_load(f"{root_path}/data/features/rxn_embed.pkl")

    for i, out in enumerate(outs): # type: ignore
        out = out.cpu().numpy().reshape(-1)  # type: ignore
        cos_sims = search_similarity(out, rxn_dict, top_k=100)

        for k, v in cos_sims:
            append_txt(f"{out_file}/pocket{i}.txt", f"{k},{F.sigmoid(torch.tensor(v * 4.603962421417236)).numpy()}\n")  # type: ignore


if __name__ == "__main__":
    for pocket_dir in os.listdir("/data/zzjun/EnzySeek/data/steroid/special/pocket/"):
        lmdb_file = f"/data/zzjun/EnzySeek/data/steroid/special/pocket/{pocket_dir}/pocket.lmdb"
        out_file = f"/data/zzjun/EnzySeek/data/steroid/special/prediction/{pocket_dir}/"
        search_rxn(lmdb_file, out_file)
