import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

from typing import Union, List, Tuple, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

# Default geometry (must satisfy: target + 2*flank <= max_seq_len)
DEFAULT_TARGET_BLOCK_SIZE = 400


class TISDataset(Dataset):
    """Dataset for Translation Initiation Site prediction.

    Implements the SpliceAI-style "positive-window mining" strategy:

    1. **Tile** each transcript into non-overlapping target blocks of
       ``target_block_size`` nucleotides.
    2. **Discard** any block that contains zero annotated TIS positions.
    3. **Expand** each surviving block with symmetric flanking context so
       the total window length equals ``max_seq_len``.  The flanking
       nucleotides provide upstream/downstream context (e.g. Kozak motifs,
       UTR structure) but are excluded from loss computation.
    4. For short transcripts (≤ ``max_seq_len``), the full sequence is
       used as a single sample and the entire sequence is the target.

    This eliminates extreme class imbalance at the dataset level: every
    training window is guaranteed to contain at least one TIS, so a
    simple shuffled DataLoader is sufficient — no structured batch
    sampler required.

    Each sample returns:
        tokens:      (L,) int64   — tokenised window (with CLS / EOS).
        labels:      (L,) float32 — 1.0 at TIS positions, 0.0 elsewhere.
        atg_mask:    (L,) bool    — True where an ATG codon begins.
        target_mask: (L,) bool    — True for positions in the target block
                                    (where loss should be computed).  False
                                    for CLS, EOS, PAD, and flanking context.
    """

    _A = "A"
    _T = "T"
    _G = "G"

    def __init__(
        self,
        csv_path: Union[str, Path],
        alphabet: Alphabet,
        max_seq_len: int = 1022,
        target_block_size: int = DEFAULT_TARGET_BLOCK_SIZE,
    ):
        super().__init__()

        self.alphabet = alphabet
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.target_block_size = target_block_size

        # Pre-resolve token indices for ATG detection
        self._a_idx = alphabet.get_idx(self._A)
        self._t_idx = alphabet.get_idx(self._T)
        self._g_idx = alphabet.get_idx(self._G)

        # Read CSV and build windows
        df = pd.read_csv(csv_path)
        # Each sample: (subsequence, tis_positions_local, target_start, target_end)
        # target_start/end are in *sequence-local* coordinates (0-based, before CLS offset)
        self.samples: List[Tuple[str, List[int], int, int]] = []
        self._build_samples(df)

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _parse_tis(self, raw) -> List[int]:
        raw = str(raw)
        if raw and raw not in ("", "nan", "None"):
            return [int(p) for p in raw.split(";") if p.strip()]
        return []

    def _build_samples(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            seq = str(row["sequence"])
            tis_positions = self._parse_tis(row["tis_positions"])
            seq_len = len(seq)

            # Short transcript: fits entirely in one window
            if self.max_seq_len is None or seq_len <= self.max_seq_len:
                if len(tis_positions) > 0:
                    self.samples.append((seq, tis_positions, 0, seq_len))
                continue

            # --- Long transcript: tile into target blocks ---------------------
            target_size = self.target_block_size
            flank = (self.max_seq_len - target_size) // 2

            for block_start in range(0, seq_len, target_size):
                block_end = min(block_start + target_size, seq_len)

                # Only keep blocks that contain at least one TIS
                block_tis = [p for p in tis_positions if block_start <= p < block_end]
                if not block_tis:
                    continue

                # Expand with flanking context
                window_start = max(0, block_start - flank)
                window_end = min(seq_len, block_end + flank)

                # If near an edge, try to fill the full max_seq_len
                window_len = window_end - window_start
                if window_len < self.max_seq_len:
                    deficit = self.max_seq_len - window_len
                    # Try expanding left first
                    if window_start > 0:
                        expand = min(deficit, window_start)
                        window_start -= expand
                        deficit -= expand
                    # Then right
                    if deficit > 0 and window_end < seq_len:
                        window_end = min(seq_len, window_end + deficit)

                # Clamp window to max_seq_len
                if window_end - window_start > self.max_seq_len:
                    window_end = window_start + self.max_seq_len

                subseq = seq[window_start:window_end]

                # Remap ALL TIS positions that fall within the window
                # (not just block_tis — a TIS in the flank still needs a label
                #  even though it won't contribute to loss via target_mask)
                sub_tis = [
                    p - window_start
                    for p in tis_positions
                    if window_start <= p < window_end
                ]

                # Target region in window-local coordinates
                target_start_local = block_start - window_start
                target_end_local = block_end - window_start

                self.samples.append((subseq, sub_tis, target_start_local, target_end_local))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tis_positions, target_start, target_end = self.samples[idx]

        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.long)
        tok_len = len(tokens)

        labels = torch.zeros(tok_len, dtype=torch.float32)
        for pos in tis_positions:
            labels[pos + 1] = 1.0  # +1 for CLS

        atg_mask = torch.zeros(tok_len, dtype=torch.bool)
        for i in range(1, tok_len - 2):
            if (
                tokens[i] == self._a_idx
                and tokens[i + 1] == self._t_idx
                and tokens[i + 2] == self._g_idx
            ):
                atg_mask[i] = True

        # Target mask: True only for positions in the target block
        # +1 offset because tokens[0] = CLS
        target_mask = torch.zeros(tok_len, dtype=torch.bool)
        target_mask[target_start + 1 : target_end + 1] = True

        return tokens, labels, atg_mask, target_mask


# ======================================================================
# Collate
# ======================================================================
def tis_collate_fn(batch):
    """Collate variable-length TIS samples into a padded batch."""
    tokens_list, labels_list, atg_list, target_list = zip(*batch)

    pad_idx = 1  # Alphabet.pad_idx

    tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_idx)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    atg_mask = pad_sequence(atg_list, batch_first=True, padding_value=False)
    target_mask = pad_sequence(target_list, batch_first=True, padding_value=False)

    return tokens, labels, atg_mask, target_mask
