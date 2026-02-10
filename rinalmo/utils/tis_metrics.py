import torch


def tis_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    atg_mask: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
):
    """Compute TIS prediction metrics including ATG-awareness diagnostics.

    Args:
        logits:      (B, L) raw logits.
        labels:      (B, L) binary ground truth.
        atg_mask:    (B, L) bool — True where ATG codon begins.
        ignore_mask: (B, L) bool — True for positions to exclude (CLS/EOS/PAD).
        threshold:   Decision threshold on sigmoid(logits).

    Returns:
        dict with keys:
            tp, fp, fn, tn              — raw counts (for epoch-level aggregation)
            non_atg_tis_tp, non_atg_tis_fn — for non-ATG TIS recall
            atg_tis_tp, atg_tis_fn         — for ATG TIS recall
            atg_neg_fp, atg_neg_tn         — for ATG specificity
    """
    preds = (torch.sigmoid(logits) >= threshold).float()

    # Build valid mask (True = include)
    if ignore_mask is not None:
        valid = ~ignore_mask
    else:
        valid = torch.ones_like(labels, dtype=torch.bool)

    p = preds[valid]
    l = labels[valid]
    atg = atg_mask[valid]

    tp = ((p == 1) & (l == 1)).sum()
    fp = ((p == 1) & (l == 0)).sum()
    fn = ((p == 0) & (l == 1)).sum()
    tn = ((p == 0) & (l == 0)).sum()

    # --- Non-ATG TIS (the ones we really don't want to miss) ---
    non_atg_tis = (l == 1) & (~atg)
    non_atg_tis_tp = ((p == 1) & non_atg_tis).sum()
    non_atg_tis_fn = ((p == 0) & non_atg_tis).sum()

    # --- ATG TIS ---
    atg_tis = (l == 1) & atg
    atg_tis_tp = ((p == 1) & atg_tis).sum()
    atg_tis_fn = ((p == 0) & atg_tis).sum()

    # --- ATG negatives (want high specificity here) ---
    atg_neg = (l == 0) & atg
    atg_neg_fp = ((p == 1) & atg_neg).sum()
    atg_neg_tn = ((p == 0) & atg_neg).sum()

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "non_atg_tis_tp": non_atg_tis_tp, "non_atg_tis_fn": non_atg_tis_fn,
        "atg_tis_tp": atg_tis_tp, "atg_tis_fn": atg_tis_fn,
        "atg_neg_fp": atg_neg_fp, "atg_neg_tn": atg_neg_tn,
    }


def aggregate_tis_metrics(accumulated: dict) -> dict:
    """Compute final metrics from accumulated counts across batches.

    Args:
        accumulated: dict whose values are scalar tensors, summed across batches.

    Returns:
        dict of computed metric values (all as Python floats, percentages).
    """
    tp = accumulated["tp"].float()
    fp = accumulated["fp"].float()
    fn = accumulated["fn"].float()
    tn = accumulated["tn"].float()

    total = tp + fp + fn + tn
    accuracy = ((tp + tn) / total * 100) if total > 0 else torch.tensor(0.0)
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else torch.tensor(0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else torch.tensor(0.0)

    # Non-ATG TIS recall
    na_tp = accumulated["non_atg_tis_tp"].float()
    na_fn = accumulated["non_atg_tis_fn"].float()
    non_atg_tis_recall = (na_tp / (na_tp + na_fn) * 100) if (na_tp + na_fn) > 0 else torch.tensor(0.0)

    # ATG TIS recall
    a_tp = accumulated["atg_tis_tp"].float()
    a_fn = accumulated["atg_tis_fn"].float()
    atg_tis_recall = (a_tp / (a_tp + a_fn) * 100) if (a_tp + a_fn) > 0 else torch.tensor(0.0)

    # ATG specificity (true-negative rate among ATG positions)
    aneg_fp = accumulated["atg_neg_fp"].float()
    aneg_tn = accumulated["atg_neg_tn"].float()
    atg_specificity = (aneg_tn / (aneg_tn + aneg_fp) * 100) if (aneg_tn + aneg_fp) > 0 else torch.tensor(0.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "non_atg_tis_recall": non_atg_tis_recall,
        "atg_tis_recall": atg_tis_recall,
        "atg_specificity": atg_specificity,
    }
