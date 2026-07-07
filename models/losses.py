import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def gather(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        return torch.cat(all_gather(x), dim=0) # type: ignore
    return x


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, prot_emb: torch.Tensor, rxn_emb: torch.Tensor) -> torch.Tensor:
        batch_size = prot_emb.size(0)

        # ensure inputs and computations are float32
        prot_emb = prot_emb.to(dtype=torch.float32)
        rxn_emb = rxn_emb.to(dtype=torch.float32)

        prot_emb = F.normalize(prot_emb, dim=-1)
        rxn_emb = F.normalize(rxn_emb, dim=-1)

        all_prot_emb = gather(prot_emb)
        all_rxn_emb = gather(rxn_emb)

        scale = self.logit_scale.exp().clamp(max=100.0)

        logits_p2r = scale * (prot_emb @ all_rxn_emb.T)
        logits_r2p = scale * (rxn_emb @ all_prot_emb.T)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            labels = torch.arange(batch_size, device=prot_emb.device) + rank * batch_size
        else:
            labels = torch.arange(batch_size, device=prot_emb.device)

        loss_p2r = F.cross_entropy(logits_p2r, labels)
        loss_r2p = F.cross_entropy(logits_r2p, labels)

        return 0.5 * (loss_p2r + loss_r2p)


class SupervisedInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, prot_emb: torch.Tensor, rxn_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # ensure inputs and computations are float32
        prot_emb = prot_emb.to(dtype=torch.float32)
        rxn_emb = rxn_emb.to(dtype=torch.float32)

        prot_emb = F.normalize(prot_emb, dim=-1)
        rxn_emb = F.normalize(rxn_emb, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (prot_emb @ rxn_emb.T)

        loss_p2r = self.masked_contrastive_loss(logits, mask)
        loss_r2p = self.masked_contrastive_loss(logits.T, mask.T)

        return 0.5 * (loss_p2r + loss_r2p)

    @staticmethod
    def masked_contrastive_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits should be float32; mask as float32 as well for targets
        logits = logits.to(dtype=torch.float32)
        mask = mask.to(device=logits.device, dtype=torch.float32)
        positive_count = mask.sum(dim=1)
        targets = mask / positive_count.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=1)
        return -(targets * log_prob).sum(dim=1).mean()


class DebiasedInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, debiasing_strength: float = 0.1):
        super().__init__()
        self.debiasing_strength = debiasing_strength
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, prot_emb, rxn_emb) -> torch.Tensor:

        # ensure inputs and computations are float32
        prot_emb = prot_emb.to(dtype=torch.float32)
        rxn_emb = rxn_emb.to(dtype=torch.float32)

        prot_emb = F.normalize(prot_emb, dim=-1)
        rxn_emb = F.normalize(rxn_emb, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (prot_emb @ rxn_emb.T)

        loss_p2r = self.directional_loss(logits, scale)
        loss_r2p = self.directional_loss(logits.T, scale)

        return 0.5 * (loss_p2r + loss_r2p)

    def directional_loss(self, similarity, scale):
        batch_size = similarity.size(0)
        num_negatives = batch_size - 1

        # ensure float32 computations
        similarity = similarity.to(dtype=torch.float32)
        scale = scale.to(dtype=torch.float32)

        positive = similarity.diagonal()
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
        relative_logits = similarity - positive[:, None]
        relative_logits = relative_logits.masked_fill(~negative_mask, -torch.inf)
        log_negative_relative = torch.logsumexp(relative_logits, dim=1)

        tau = float(self.debiasing_strength)
        # convert scalar corrections to tensors with matching dtype/device
        log_correction = torch.tensor(math.log(tau * num_negatives), dtype=similarity.dtype, device=similarity.device)
        valid = log_negative_relative > log_correction
        delta = log_correction - log_negative_relative
        safe_delta = torch.where(valid, delta, torch.full_like(delta, -1.0))

        log_subtracted = log_negative_relative + torch.log1p(-torch.exp(safe_delta))
        log_subtracted = torch.where(valid, log_subtracted, torch.full_like(log_subtracted, -torch.inf))

        log_debiased_negative = log_subtracted - torch.tensor(math.log1p(-tau), dtype=similarity.dtype, device=similarity.device)
        log_lower_bound = torch.tensor(math.log(num_negatives), dtype=similarity.dtype, device=similarity.device) - scale - positive
        log_debiased_negative = torch.maximum(log_debiased_negative, log_lower_bound)

        return F.softplus(log_debiased_negative).mean()


class TopInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, top_pct: float = 0.05):
        super().__init__()
        self.top_pct = top_pct
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, prot_emb, rxn_emb) -> torch.Tensor:

        # ensure inputs and computations are float32
        prot_emb = prot_emb.to(dtype=torch.float32)
        rxn_emb = rxn_emb.to(dtype=torch.float32)

        prot_emb = F.normalize(prot_emb, dim=-1)
        rxn_emb = F.normalize(rxn_emb, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (prot_emb @ rxn_emb.T)

        loss_p2r = self.directional_loss(logits)
        loss_r2p = self.directional_loss(logits.T)

        return 0.5 * (loss_p2r + loss_r2p)

    def directional_loss(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        # ensure float32 computations
        logits = logits.to(dtype=torch.float32)

        num_ignore = min(int((batch_size - 1) * self.top_pct), batch_size - 2)

        # if there is nothing to ignore, just compute standard cross-entropy
        if num_ignore <= 0:
            labels = torch.arange(batch_size, device=logits.device)
            return F.cross_entropy(logits, labels)

        selection = logits.detach().clone()
        selection.fill_diagonal_(-torch.inf)
        ignore_indices = selection.topk(num_ignore, dim=1, sorted=False).indices

        logits = logits.scatter(1, ignore_indices, -torch.inf)
        labels = torch.arange(batch_size, device=logits.device)

        return F.cross_entropy(logits, labels)


class ContrastiveLoss(nn.Module):
    # 包装器
    def __init__(self, temperature: float = 0.07):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.alpha = 0.7  # weight for the contrastive loss component

    def forward(self, prot_emb: torch.Tensor, rxn_emb: torch.Tensor) -> torch.Tensor:
        batch_size = prot_emb.size(0)

        # ensure inputs and computations are float32
        prot_emb = prot_emb.to(dtype=torch.float32)
        rxn_emb = rxn_emb.to(dtype=torch.float32)

        prot_emb = F.normalize(prot_emb, dim=-1)
        rxn_emb = F.normalize(rxn_emb, dim=-1)

        all_prot_emb = gather(prot_emb)
        all_rxn_emb = gather(rxn_emb)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_p2r = scale * (prot_emb @ all_rxn_emb.T)
        logits_r2p = scale * (rxn_emb @ all_prot_emb.T)
        logits_p2p = scale * prot_emb @ all_prot_emb.T
        logits_r2r = scale * rxn_emb @ all_rxn_emb.T

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            labels = torch.arange(batch_size, device=prot_emb.device) + rank * batch_size
        else:
            labels = torch.arange(batch_size, device=prot_emb.device)

        loss_p2r = F.cross_entropy(logits_p2r, labels)
        loss_r2p = F.cross_entropy(logits_r2p, labels)

        clip_loss = 0.5 * (loss_p2r + loss_r2p)
        kl_loss = 0.5 * (
            self.kl_loss(F.log_softmax(logits_p2r, dim=1), F.softmax(logits_p2p.detach(), dim=1))
            + self.kl_loss(F.log_softmax(logits_r2p, dim=1), F.softmax(logits_r2r.detach(), dim=1))
        )

        return self.alpha * clip_loss + (1 - self.alpha) * kl_loss
