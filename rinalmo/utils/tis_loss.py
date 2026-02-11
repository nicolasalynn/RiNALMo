import torch
from torch import nn


class CompoundTISLoss(nn.Module):
    """Compound loss for TIS prediction: Tversky + ATG-contrastive BCE.

    Two complementary terms:

    1. **Tversky loss** — a soft, differentiable generalisation of the Dice
       coefficient that directly optimises TP / (TP + α·FP + β·FN).
       Setting β > α biases toward recall, which is what we want: missing a
       real TIS is worse than a false alarm.  Critically, Tversky loss is
       *insensitive to true negatives*, so the sea of non-TIS positions
       does not dominate the gradient.

    2. **ATG-contrastive BCE** — binary cross-entropy computed *only* on
       positions where an ATG codon begins.  This forces the model to
       explicitly discriminate TIS-ATGs from non-TIS-ATGs rather than
       learning the shortcut "ATG → positive".

    Args:
        tversky_alpha: Weight on false positives in Tversky denominator.
        tversky_beta:  Weight on false negatives (β > α → recall bias).
        atg_lambda:    Mixing coefficient for the ATG-contrastive term.
        smooth:        Smoothing constant to avoid division by zero.
    """

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        atg_lambda: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = tversky_alpha
        self.beta = tversky_beta
        self.atg_lambda = atg_lambda
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        atg_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:      (B, L) raw logits per position.
            labels:      (B, L) binary targets.
            atg_mask:    (B, L) bool — True where ATG codon begins.
            target_mask: (B, L) bool — True for target-block positions
                         (where loss should be computed).

        Returns:
            Scalar loss.
        """
        probs = torch.sigmoid(logits)

        # ---- Tversky loss (on target positions only) ---------------------
        p = probs[target_mask]
        t = labels[target_mask]

        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tversky_loss = 1.0 - tversky_index

        # ---- ATG-contrastive BCE (only on ATG positions in target) --------
        atg_valid = target_mask & atg_mask
        if atg_valid.any():
            atg_logits = logits[atg_valid]
            atg_labels = labels[atg_valid]
            atg_bce = nn.functional.binary_cross_entropy_with_logits(
                atg_logits, atg_labels, reduction="mean"
            )
        else:
            atg_bce = torch.tensor(0.0, device=logits.device)

        return tversky_loss + self.atg_lambda * atg_bce
