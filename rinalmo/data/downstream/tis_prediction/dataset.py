import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import random
import math

from typing import Union, List, Tuple, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

# Window categories for structured batch construction
CAT_POSITIVE = 0   # Contains at least one TIS site
CAT_HARD_NEG = 1   # No TIS, but contains ATG codons (confounders)
CAT_EASY_NEG = 2   # No TIS, no ATG codons


class TISDataset(Dataset):
    """Dataset for Translation Initiation Site prediction.

    Expects a CSV with columns:
        - sequence:      The mRNA/RNA nucleotide sequence.
        - tis_positions: Semicolon-separated 0-based positions in the sequence
                         that are annotated translation initiation sites.
                         Empty or "nan" means no TIS in this sequence.

    An optional ``transcript_id`` column is ignored during training but kept
    in the DataFrame for traceability.

    Windowing
    ---------
    Because RiNALMo was pre-trained on sequences up to ~1022 nt (1024 tokens
    with CLS/EOS), longer transcripts are split into windows at dataset-
    construction time.

    Each window is categorised as:
      - **positive**: contains at least one annotated TIS site
      - **hard_neg**: no TIS but contains ATG codons (the confounders the
        model must learn to reject)
      - **easy_neg**: no TIS and no ATG codons

    These categories enable structured batch construction via
    :class:`StructuredTISBatchSampler`.

    Each sample returns:
        tokens:      (L,) int64   — tokenised window (with CLS / EOS).
        labels:      (L,) float32 — 1.0 at TIS positions, 0.0 elsewhere.
        atg_mask:    (L,) bool    — True where an ATG codon begins.
        ignore_mask: (L,) bool    — True for CLS, EOS, PAD (loss-excluded).
    """

    # Token characters for ATG detection (alphabet maps U→T internally)
    _A = "A"
    _T = "T"
    _G = "G"

    def __init__(
        self,
        csv_path: Union[str, Path],
        alphabet: Alphabet,
        max_seq_len: int = 1022,
        neg_window_stride: Optional[int] = None,
    ):
        super().__init__()

        self.alphabet = alphabet
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.neg_window_stride = neg_window_stride or (
            max_seq_len // 2 if max_seq_len else None
        )

        # Pre-resolve token indices for ATG detection
        self._a_idx = alphabet.get_idx(self._A)
        self._t_idx = alphabet.get_idx(self._T)
        self._g_idx = alphabet.get_idx(self._G)

        # Read CSV and pre-compute windows
        df = pd.read_csv(csv_path)
        self.samples: List[Tuple[str, List[int]]] = []
        self.categories: List[int] = []  # CAT_POSITIVE / CAT_HARD_NEG / CAT_EASY_NEG
        self._build_samples(df)

        # Build per-category index lists for the structured sampler
        self.positive_indices: List[int] = []
        self.hard_neg_indices: List[int] = []
        self.easy_neg_indices: List[int] = []
        for i, cat in enumerate(self.categories):
            if cat == CAT_POSITIVE:
                self.positive_indices.append(i)
            elif cat == CAT_HARD_NEG:
                self.hard_neg_indices.append(i)
            else:
                self.easy_neg_indices.append(i)

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _parse_tis(self, raw) -> List[int]:
        raw = str(raw)
        if raw and raw not in ("", "nan", "None"):
            return [int(p) for p in raw.split(";") if p.strip()]
        return []

    def _seq_has_atg(self, seq: str) -> bool:
        """Check if a nucleotide string contains an ATG triplet."""
        return "ATG" in seq.upper().replace("U", "T")

    def _clamp_window(self, centre: int, seq_len: int) -> Tuple[int, int]:
        half = self.max_seq_len // 2
        start = max(0, centre - half)
        end = start + self.max_seq_len
        if end > seq_len:
            end = seq_len
            start = max(0, end - self.max_seq_len)
        return start, end

    def _categorise(self, seq: str, tis_positions: List[int]) -> int:
        if len(tis_positions) > 0:
            return CAT_POSITIVE
        elif self._seq_has_atg(seq):
            return CAT_HARD_NEG
        else:
            return CAT_EASY_NEG

    def _build_samples(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            seq = str(row["sequence"])
            tis_positions = self._parse_tis(row["tis_positions"])
            seq_len = len(seq)

            if self.max_seq_len is None or seq_len <= self.max_seq_len:
                self.samples.append((seq, tis_positions))
                self.categories.append(self._categorise(seq, tis_positions))
                continue

            # --- Windowing for long sequences ----------------------------------
            windows_used: List[Tuple[int, int]] = []

            for pos in tis_positions:
                start, end = self._clamp_window(pos, seq_len)
                windows_used.append((start, end))

            stride = self.neg_window_stride
            start = 0
            while start < seq_len:
                end = min(start + self.max_seq_len, seq_len)
                if not any(ws == start and we == end for ws, we in windows_used):
                    windows_used.append((start, end))
                start += stride
                if end == seq_len:
                    break

            seen = set()
            for w_start, w_end in windows_used:
                if (w_start, w_end) in seen:
                    continue
                seen.add((w_start, w_end))

                subseq = seq[w_start:w_end]
                sub_tis = [
                    p - w_start
                    for p in tis_positions
                    if w_start <= p < w_end
                ]
                self.samples.append((subseq, sub_tis))
                self.categories.append(self._categorise(subseq, sub_tis))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tis_positions = self.samples[idx]

        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.long)
        tok_len = len(tokens)

        labels = torch.zeros(tok_len, dtype=torch.float32)
        for pos in tis_positions:
            labels[pos + 1] = 1.0

        atg_mask = torch.zeros(tok_len, dtype=torch.bool)
        for i in range(1, tok_len - 2):
            if (
                tokens[i] == self._a_idx
                and tokens[i + 1] == self._t_idx
                and tokens[i + 2] == self._g_idx
            ):
                atg_mask[i] = True

        ignore_mask = torch.zeros(tok_len, dtype=torch.bool)
        ignore_mask[0] = True
        ignore_mask[len(seq) + 1] = True

        return tokens, labels, atg_mask, ignore_mask


# ======================================================================
# Structured batch sampler
# ======================================================================
class StructuredTISBatchSampler(Sampler):
    """Constructs batches with a controlled mix of window categories.

    Each batch of size B contains:
      - ``pos_frac * B``  positive windows  (TIS-containing)
      - ``hard_frac * B`` hard negatives     (ATG but no TIS)
      - remainder         easy negatives     (no ATG, no TIS)

    Within each category, samples are drawn randomly without replacement
    until the pool is exhausted, then reshuffled (infinite epoch).
    This means the model sees each positive window roughly the same number
    of times per epoch, but batches are always balanced.

    Args:
        dataset:   A TISDataset instance.
        batch_size: Samples per batch.
        pos_frac:  Fraction of batch that should be positive windows.
        hard_frac: Fraction of batch that should be hard negatives.
        epoch_len: Number of batches per epoch.  Defaults to
                   ``len(dataset) // batch_size``.
    """

    def __init__(
        self,
        dataset: TISDataset,
        batch_size: int,
        pos_frac: float = 0.5,
        hard_frac: float = 0.3,
        epoch_len: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.pos_frac = pos_frac
        self.hard_frac = hard_frac
        self.epoch_len = epoch_len or (len(dataset) // batch_size)

        self._pos = list(dataset.positive_indices)
        self._hard = list(dataset.hard_neg_indices)
        self._easy = list(dataset.easy_neg_indices)

        # Fallback: if a category is empty, redistribute to others
        if not self._hard:
            self._hard = self._easy
        if not self._easy:
            self._easy = self._hard

    def _infinite_shuffle(self, pool: List[int]):
        """Yield indices from *pool* forever, reshuffling each pass."""
        buf = []
        while True:
            if not buf:
                buf = pool.copy()
                random.shuffle(buf)
            yield buf.pop()

    def __iter__(self):
        n_pos = max(1, round(self.batch_size * self.pos_frac))
        n_hard = max(1, round(self.batch_size * self.hard_frac))
        n_easy = self.batch_size - n_pos - n_hard

        pos_gen = self._infinite_shuffle(self._pos)
        hard_gen = self._infinite_shuffle(self._hard)
        easy_gen = self._infinite_shuffle(self._easy)

        for _ in range(self.epoch_len):
            batch = []
            batch.extend(next(pos_gen) for _ in range(n_pos))
            batch.extend(next(hard_gen) for _ in range(n_hard))
            batch.extend(next(easy_gen) for _ in range(n_easy))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.epoch_len


# ======================================================================
# Collate
# ======================================================================
def tis_collate_fn(batch):
    """Collate variable-length TIS samples into a padded batch."""
    tokens_list, labels_list, atg_list, ignore_list = zip(*batch)

    pad_idx = 1  # Alphabet.pad_idx

    tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_idx)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    atg_mask = pad_sequence(atg_list, batch_first=True, padding_value=False)
    ignore_mask = pad_sequence(ignore_list, batch_first=True, padding_value=True)

    return tokens, labels, atg_mask, ignore_mask
