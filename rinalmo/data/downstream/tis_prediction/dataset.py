import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import random

from typing import Union, List, Tuple, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet


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

    For transcripts longer than ``max_seq_len``:
      - One window is centred on each annotated TIS position (so no TIS site
        is ever lost to truncation).
      - Additional random windows are sampled to provide negative context.
        The number of random windows scales with transcript length so that
        coverage is roughly proportional.
      - Windows at transcript boundaries are clamped (not wrapped).

    For transcripts that fit within ``max_seq_len``, the full sequence is
    used as a single sample — no windowing is applied.

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
        """
        Args:
            csv_path:          Path to the CSV file.
            alphabet:          RiNALMo Alphabet instance.
            max_seq_len:       Maximum nucleotide length per window (excluding
                               CLS/EOS tokens).  Set to 0 or None to disable
                               windowing entirely.
            neg_window_stride: For long transcripts without any TIS, stride
                               between consecutive negative windows.  Defaults
                               to ``max_seq_len // 2`` (50% overlap).
        """
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
        self.samples: List[Tuple[str, List[int]]] = []  # (subseq, tis_positions_in_subseq)
        self._build_samples(df)

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _parse_tis(self, raw) -> List[int]:
        raw = str(raw)
        if raw and raw not in ("", "nan", "None"):
            return [int(p) for p in raw.split(";") if p.strip()]
        return []

    def _clamp_window(self, centre: int, seq_len: int) -> Tuple[int, int]:
        """Return (start, end) of a window centred on *centre*, clamped to [0, seq_len)."""
        half = self.max_seq_len // 2
        start = max(0, centre - half)
        end = start + self.max_seq_len
        if end > seq_len:
            end = seq_len
            start = max(0, end - self.max_seq_len)
        return start, end

    def _build_samples(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            seq = str(row["sequence"])
            tis_positions = self._parse_tis(row["tis_positions"])
            seq_len = len(seq)

            # No windowing needed
            if self.max_seq_len is None or seq_len <= self.max_seq_len:
                self.samples.append((seq, tis_positions))
                continue

            # --- Windowing for long sequences ----------------------------------
            windows_used: List[Tuple[int, int]] = []

            # 1) One window centred on each TIS site
            for pos in tis_positions:
                start, end = self._clamp_window(pos, seq_len)
                windows_used.append((start, end))

            # 2) Strided negative windows for coverage
            stride = self.neg_window_stride
            start = 0
            while start < seq_len:
                end = min(start + self.max_seq_len, seq_len)
                # Only keep if it doesn't duplicate a TIS-centred window
                if not any(ws == start and we == end for ws, we in windows_used):
                    windows_used.append((start, end))
                start += stride
                if end == seq_len:
                    break

            # Deduplicate (same start,end from different TIS that map to same clamp)
            seen = set()
            for w_start, w_end in windows_used:
                if (w_start, w_end) in seen:
                    continue
                seen.add((w_start, w_end))

                subseq = seq[w_start:w_end]
                # Re-map TIS positions that fall inside this window
                sub_tis = [
                    p - w_start
                    for p in tis_positions
                    if w_start <= p < w_end
                ]
                self.samples.append((subseq, sub_tis))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tis_positions = self.samples[idx]

        # --- Tokenise ---------------------------------------------------------
        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.long)
        tok_len = len(tokens)  # CLS + seq + EOS

        # --- Labels (aligned with token positions; +1 offset for CLS) ---------
        labels = torch.zeros(tok_len, dtype=torch.float32)
        for pos in tis_positions:
            labels[pos + 1] = 1.0  # +1 because tokens[0] = CLS

        # --- ATG mask ---------------------------------------------------------
        atg_mask = torch.zeros(tok_len, dtype=torch.bool)
        for i in range(1, tok_len - 2):  # only within actual sequence
            if (
                tokens[i] == self._a_idx
                and tokens[i + 1] == self._t_idx
                and tokens[i + 2] == self._g_idx
            ):
                atg_mask[i] = True

        # --- Ignore mask (CLS=0, EOS=last-non-pad) ---------------------------
        ignore_mask = torch.zeros(tok_len, dtype=torch.bool)
        ignore_mask[0] = True            # CLS
        ignore_mask[len(seq) + 1] = True  # EOS

        return tokens, labels, atg_mask, ignore_mask


def tis_collate_fn(batch):
    """Collate variable-length TIS samples into a padded batch.

    Pads tokens with pad_idx=1 (the Alphabet default) and labels / masks
    with 0 / False / True (for ignore_mask) respectively.
    """
    tokens_list, labels_list, atg_list, ignore_list = zip(*batch)

    pad_idx = 1  # Alphabet.pad_idx

    tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_idx)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    atg_mask = pad_sequence(atg_list, batch_first=True, padding_value=False)
    # Padded positions should be ignored in loss
    ignore_mask = pad_sequence(ignore_list, batch_first=True, padding_value=True)

    return tokens, labels, atg_mask, ignore_mask
