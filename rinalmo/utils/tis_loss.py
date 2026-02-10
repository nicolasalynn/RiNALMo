import torch
from torch import nn


class TISAwareFocalLoss(nn.Module):
    """Focal loss with ATG-aware sample weighting for TIS prediction.

    Addresses two key challenges:
      1. Class imbalance — very few TIS positions vs. many non-TIS positions.
      2. ATG bias — most annotated TIS are ATG/AUG, risking a trivial classifier
         that simply flags every ATG. We counteract this by:
         - Up-weighting non-ATG TIS sites (so the model must learn to find them).
         - Up-weighting ATG non-TIS sites (so the model learns ATG != TIS).

    Args:
        gamma: Focal-loss focusing parameter. Higher values down-weight easy
            examples more aggressively (default 2.0).
        pos_weight: Base weight for all positive (TIS) positions to compensate
            for class imbalance.
        non_atg_tis_bonus: Extra multiplier applied *on top of* pos_weight for
            TIS positions that are NOT the start of an ATG codon.
        atg_neg_weight: Weight for ATG positions that are NOT TIS — these are
            the critical hard negatives the model must learn to reject.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 10.0,
        non_atg_tis_bonus: float = 5.0,
        atg_neg_weight: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.non_atg_tis_bonus = non_atg_tis_bonus
        self.atg_neg_weight = atg_neg_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        atg_mask: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:      (B, L) raw logits per token position.
            labels:      (B, L) binary labels (1 = TIS, 0 = not TIS).
            atg_mask:    (B, L) bool — True where an ATG codon begins.
            ignore_mask: (B, L) bool — True for positions to exclude from loss
                         (CLS, EOS, PAD tokens).

        Returns:
            Scalar loss averaged over valid positions.
        """
        probs = torch.sigmoid(logits)

        # ---- focal modulation ------------------------------------------------
        p_t = labels * probs + (1.0 - labels) * (1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma

        # ---- per-element BCE (numerically stable via log-sum-exp) ------------
        # Using the identity: BCE = max(logits, 0) - logits*labels + log(1+exp(-|logits|))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )

        # ---- class-balance weights -------------------------------------------
        class_weight = torch.where(labels == 1.0, self.pos_weight, 1.0)

        # ---- ATG-aware weights -----------------------------------------------
        atg_aware_weight = torch.ones_like(labels)
        # Non-ATG TIS: rare and critical to detect
        atg_aware_weight = torch.where(
            (labels == 1.0) & (~atg_mask), self.non_atg_tis_bonus, atg_aware_weight
        )
        # ATG non-TIS: hard negatives — model must learn to reject these
        atg_aware_weight = torch.where(
            (labels == 0.0) & atg_mask, self.atg_neg_weight, atg_aware_weight
        )

        # ---- combine ---------------------------------------------------------
        loss = focal_weight * class_weight * atg_aware_weight * bce

        # ---- mask out ignored positions (CLS, EOS, PAD) ----------------------
        if ignore_mask is not None:
            loss = loss.masked_fill(ignore_mask, 0.0)
            n_valid = (~ignore_mask).sum().clamp(min=1)
        else:
            n_valid = loss.numel()

        return loss.sum() / n_valid
