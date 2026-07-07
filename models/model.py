from models.losses import *
from models.operations import *
from models.layers import PocEnc, RXNEnc
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.adamw import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
import torchmetrics as tm
from utils.general import pkl_load

###############################################################
def calc_diagonal_sim(prot_emb, rxn_emb):
    prot_emb = F.normalize(prot_emb, dim=-1)
    rxn_emb = F.normalize(rxn_emb, dim=-1)

    return (prot_emb * rxn_emb).sum(dim=-1)

##############################################################
class TaskModel(pl.LightningModule):
    def __init__(
        self,
        total_steps: int = 10000,
        lr: float = 1e-3,
        lr_warmup_steps: int = 100,
        weight_decay: float = 1e-3,
        loss_name: str = "infonce",
        **kwargs,  # for compatibility with other configs
    ):
        super().__init__()
        self.total_steps = total_steps
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self._model()
        self._loss_fn()
        self._pair()
        self._metrics()

    def forward(self, graph):
        poc, rxn = graph
        poc_emb = self.poc_encoder(poc)
        rxn_emb = self.rxn_encoder(rxn)
        if self.loss_name == "supervised_infonce":
            mask = [[1.0 if (p_id, r_id) in self.pos_pair_set else 0.0 for r_id in rxn.id] for p_id in poc.id]
            return poc_emb, rxn_emb, torch.tensor(mask, dtype=torch.float32, device=poc_emb.device)
        else:
            return poc_emb, rxn_emb

    def training_step(self, batch, batch_idx):
        if self.loss_name == "supervised_infonce":
            poc_emb, rxn_emb, mask = self(batch)
            loss = self.loss_fn(poc_emb, rxn_emb, mask)
        else:
            poc_emb, rxn_emb = self(batch)
            loss = self.loss_fn(poc_emb, rxn_emb)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=poc_emb.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        if self.loss_name == "supervised_infonce":
            poc_emb, rxn_emb, mask = self(batch)
            loss = self.loss_fn(poc_emb, rxn_emb, mask)
        else:
            poc_emb, rxn_emb = self(batch)
            loss = self.loss_fn(poc_emb, rxn_emb)

        # self.valid_metrics["diag_sim"].update(calc_diagonal_sim(poc_emb, rxn_emb))
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=poc_emb.size(0))
        return loss

    def on_validation_epoch_end(self):
        # self.log_dict(self.valid_metrics.compute(), prog_bar=True, sync_dist=True)
        # self.valid_metrics.reset()
        pass

    def predict_step(self, batch, batch_idx):
        if batch[0].type == "rxn":
            rxn_emb = self.rxn_encoder(batch)
            return rxn_emb
        elif batch[0].type == "poc":
            poc_emb = self.poc_encoder(batch)
            return poc_emb

    def configure_optimizers(self):  # type: ignore
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if param.ndim < 2 or name.endswith(".bias") or param.ndim <= 1 or "logit_scale" in name or "temperature" in name or "norm" in name_lower:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW([{"params": decay_params, "weight_decay": self.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}], lr=self.lr)
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, self.lr_warmup_steps, self.total_steps, 0.5, min_lr_rate=0.01)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "strict": True}
        return [optimizer], [scheduler]

    def _model(self):
        self.poc_encoder = PocEnc(**dict(OmegaConf.load(f"../configs/model.yaml")))
        self.rxn_encoder = RXNEnc(**dict(OmegaConf.load(f"../configs/model.yaml")))
        self.apply(weights_init)
        

    def _loss_fn(self):
        if self.loss_name == "infonce":
            self.loss_fn = InfoNCELoss(0.07)
        elif self.loss_name == "supervised_infonce":
            self.loss_fn = SupervisedInfoNCELoss(0.07)
        elif self.loss_name == "debiased_infonce":
            self.loss_fn = DebiasedInfoNCELoss(0.07, 0.01)
        elif self.loss_name == "top_infonce":
            self.loss_fn = TopInfoNCELoss(0.07, 0.01)
        elif self.loss_name == "hybrid":
            self.loss_fn = ContrastiveLoss(0.07)

    def _pair(self):
        if self.loss_name == "supervised_infonce":
            self.pos_pair_set = set(pkl_load(f"{root_path}/data/training/clip_all.pkl"))

    def _metrics(self):
        metrics = tm.MetricCollection(
            {
                "diag_sim": tm.MeanMetric(),
            }
        )
        self.valid_metrics = metrics.clone(prefix="valid/")
