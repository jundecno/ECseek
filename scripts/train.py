import lightning
import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from utils import pkl_load
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from models.model import TaskModel
from datetime import datetime
from torch_geometric.data.lightning import LightningDataset
from models.dataloader import TrainDataset
import lightning.pytorch as pl
from pathlib import Path


def train_on_rhea_with_clip(train_pairs_file, valid_pairs_file=None):
    cfg = OmegaConf.load("../configs/train.yaml")
    callbacks = OmegaConf.load("../configs/callbacks.yaml")
    loggers = OmegaConf.load("../configs/loggers.yaml")
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    dataset_type = cfg.dataset
    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_pairs = pkl_load(train_pairs_file)
    train_dataset = TrainDataset(f"{root_path}/data/features", train_pairs)

    if valid_pairs_file is not None:
        valid_pairs = pkl_load(valid_pairs_file)
        valid_dataset = TrainDataset(f"{root_path}/data/features", valid_pairs)
        datamodule = LightningDataset(
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            batch_size=cfg.batch_size,
            drop_last=True,
            num_workers=cfg.num_workers,
            follow_batch=["mol_cls"],
        )
    else:
        datamodule = LightningDataset(
            train_dataset=train_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers, follow_batch=["mol_cls"]
        )

    early_stopping = instantiate(callbacks.early_stopping)
    ckpt_save = instantiate(callbacks.model_checkpoint, dirpath=f"{root_path}/checkpoints/{dataset_type}/{now_time}/")
    total_steps = (cfg.num_epochs * len(train_dataset)) // (cfg.batch_size * len(cfg.devices))
    train_model = TaskModel(**dict(cfg), lr_warmup_steps=total_steps // 10, total_steps=total_steps)

    logger = instantiate(loggers.wandb, name=now_time)
    logger.log_hyperparams(cfg)
    logger.experiment.log_code(root=f"{root_path}/models", include_fn=lambda path, root: path.endswith(".py"))
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        devices=cfg.devices,
        accelerator="gpu",
        precision="16-mixed",
        logger=logger,
        callbacks=[early_stopping, ckpt_save],
    )
    trainer.fit(train_model, datamodule=datamodule)


if __name__ == "__main__":
    # train_on_rhea_with_clip(f"{root_path}/data/training/clip_train.pkl", f"{root_path}/data/training/clip_valid.pkl")
    train_on_rhea_with_clip(f"{root_path}/data/training/clip_all.pkl")
    # train_on_rhea_with_clip_cv(f"{root_path}/data/training/clip_all.pkl", n_splits=5)
