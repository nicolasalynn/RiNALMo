import torch
import torch.nn as nn
from typing import List


class DilatedResNet1DBlock(nn.Module):
    """Dilated 1D residual block: two dilated Conv1d layers with BatchNorm, ELU, and dropout."""

    def __init__(self, channels: int, kernel_size: int = 9, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv_net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x: (B, C, L)
        return x + self.conv_net(x)


class DilatedConvTISModel(nn.Module):
    """Standalone SpliceAI-style TIS prediction model.

    Dilated 1D convolutions for local/medium-range patterns followed by
    Transformer encoder layers for long-range context.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        padding_idx: int = 1,
        conv_channels: int = 256,
        kernel_size: int = 9,
        dilations: List[int] = None,
        dropout: float = 0.1,
        num_transformer_layers: int = 2,
        transformer_heads: int = 8,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]

        self.embedding = nn.Embedding(vocab_size, conv_channels, padding_idx=padding_idx)

        self.conv_blocks = nn.ModuleList([
            DilatedResNet1DBlock(conv_channels, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=transformer_heads,
            dim_feedforward=conv_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.output_conv = nn.Conv1d(conv_channels, 1, kernel_size=1)

    def forward(self, tokens):
        # tokens: (B, L) integer token IDs
        x = self.embedding(tokens)  # (B, L, C)
        x = x.permute(0, 2, 1)     # (B, C, L)

        for block in self.conv_blocks:
            x = block(x)

        x = x.permute(0, 2, 1)     # (B, L, C)
        x = self.transformer(x)    # (B, L, C)
        x = x.permute(0, 2, 1)     # (B, C, L)

        logits = self.output_conv(x).squeeze(1)  # (B, L)
        return logits
