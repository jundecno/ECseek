from models.operations import *
from models.layers import PocEnc, RXNEnc
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.adamw import AdamW
import torchmetrics as tm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


class TaskModel(pl.LightningModule):
    def __init__(
        self,
        total_steps: int = 10000,
        lr: float = 1e-3,
        lr_warmup_steps: int = 100,
        weight_decay: float = 1e-3,
        **kwargs,  # for compatibility with other configs
    ):
        super().__init__()
        self.total_steps = total_steps
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.weight_decay = weight_decay
        self._model()
        self._loss_fn()

    def forward(self, graph):
        poc, pos_rxn, neg_rxn = graph
        poc_emb = self.poc_encoder(poc)
        pos_rxn_emb = self.rxn_encoder(pos_rxn)
        neg_rxn_emb = self.rxn_encoder(neg_rxn)
        return poc_emb, pos_rxn_emb, neg_rxn_emb

    def training_step(self, batch, batch_idx):
        poc_emb, pos_rxn_emb, neg_rxn_emb = self(batch)
        triplet_loss = self.triplet_loss(poc_emb, pos_rxn_emb, neg_rxn_emb)
        self.log("train/loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=poc_emb.size(0))
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        poc_emb, pos_rxn_emb, neg_rxn_emb = self(batch)
        triplet_loss = self.triplet_loss(poc_emb, pos_rxn_emb, neg_rxn_emb)
        self.log("valid/loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=poc_emb.size(0))
        return triplet_loss

    def predict_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):  # type: ignore
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=1e-8,
        )
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, self.lr_warmup_steps, self.total_steps, 0.5, min_lr=self.lr * 0.1)
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def _model(self):
        self.poc_encoder = PocEnc(**dict(OmegaConf.load(f"../configs/model.yaml")))
        self.poc_encoder.apply(weights_init)
        self.rxn_encoder = RXNEnc(**dict(OmegaConf.load(f"../configs/model.yaml")))
        self.rxn_encoder.apply(weights_init)

    def _loss_fn(self):
        # 对比学习
        self.triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
