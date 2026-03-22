import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import pkl_load
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.model import TaskModel, set_all_seed
from datetime import datetime

from models.dataloader import RHEADataset, DataLoader
import lightning.pytorch as pl


def train_on_rhea(train_pairs_file, valid_pairs_file):
    cfg = OmegaConf.load("../configs/train.yaml")
    callbacks = OmegaConf.load("../configs/callbacks.yaml")
    loggers = OmegaConf.load("../configs/loggers.yaml")
    torch.set_float32_matmul_precision("high")
    set_all_seed(cfg.seed)
    dataset_type = cfg.dataset
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_pairs = pkl_load(train_pairs_file)
    valid_pairs = pkl_load(valid_pairs_file)
    train_dataset = RHEADataset(f"{root_path}/data/features", train_pairs)
    valid_dataset = RHEADataset(f"{root_path}/data/features", valid_pairs)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, follow_batch=["seq"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, follow_batch=["seq"]
    )

    early_stopping = instantiate(callbacks.early_stopping)
    ckpt_save = instantiate(callbacks.model_checkpoint, dirpath=f"{root_path}/{dataset_type}/{now_time}/")
    total_steps = (cfg.num_epochs * len(train_dataset)) // (cfg.batch_size * len(cfg.devices))
    train_model = TaskModel(**dict(cfg), lr_warmup_steps=total_steps // 10, total_steps=total_steps)
    logger = instantiate(loggers.wandb, name=now_time)
    logger.log_hyperparams(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        devices=cfg.devices,
        accelerator="gpu",
        logger=logger,
        callbacks=[early_stopping, ckpt_save],
    )
    trainer.fit(train_model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    train_on_rhea(f"{root_path}/data/training/rhea_train_pair.pkl", f"{root_path}/data/training/rhea_valid_pair.pkl")
