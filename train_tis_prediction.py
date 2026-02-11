import torch
import torch.nn as nn
from torch.optim import AdamW

import pytorch_lightning as pl

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

import argparse
from pathlib import Path
from datetime import timedelta

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.tis_prediction.datamodule import TISDataModule
from rinalmo.model.model import RiNALMo
from rinalmo.model.downstream import TISPredictionHead
from rinalmo.model.dilated_conv import DilatedConvTISModel
from rinalmo.config import model_config
from rinalmo.utils.tis_loss import CompoundTISLoss
from rinalmo.utils.tis_metrics import tis_binary_metrics, aggregate_tis_metrics, tis_topk_precision

PRED_HEAD_EMBED_DIM = 128
PRED_HEAD_NUM_BLOCKS = 2
PRED_HEAD_KERNEL_SIZE = 9


class TISPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        architecture: str = "rinalmo",
        lm_config: str = "giga",
        head_embed_dim: int = PRED_HEAD_EMBED_DIM,
        head_num_blocks: int = PRED_HEAD_NUM_BLOCKS,
        head_kernel_size: int = PRED_HEAD_KERNEL_SIZE,
        head_dropout: float = 0.1,
        finetune_lm: bool = False,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        # Loss hyper-parameters
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        atg_lambda: float = 1.0,
        # Dilated-conv architecture hyper-parameters
        conv_channels: int = 256,
        conv_kernel_size: int = 9,
        conv_dilations: list = None,
        conv_dropout: float = 0.1,
        num_transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.architecture = architecture

        if architecture == "dilated_conv":
            self.model = DilatedConvTISModel(
                conv_channels=conv_channels,
                kernel_size=conv_kernel_size,
                dilations=conv_dilations,
                dropout=conv_dropout,
                num_transformer_layers=num_transformer_layers,
                transformer_heads=transformer_heads,
            )
        else:
            self.rinalmo = RiNALMo(model_config(lm_config))
            self.pred_head = TISPredictionHead(
                c_in=self.rinalmo.config["model"]["transformer"].embed_dim,
                embed_dim=head_embed_dim,
                num_blocks=head_num_blocks,
                kernel_size=head_kernel_size,
                dropout=head_dropout,
            )
            self.finetune_lm = finetune_lm
            # Freeze the pretrained LM by default — only train the head
            if not finetune_lm:
                for param in self.rinalmo.parameters():
                    param.requires_grad = False

        self.loss_fn = CompoundTISLoss(
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            atg_lambda=atg_lambda,
        )
        self.lr = lr
        self.weight_decay = weight_decay

        self.val_step_outputs = []

    def load_pretrained_rinalmo_weights(self, pretrained_weights_path):
        if self.architecture == "dilated_conv":
            return  # No pretrained LM to load for standalone model
        self.rinalmo.load_state_dict(torch.load(pretrained_weights_path), strict=False)

    def forward(self, tokens):
        if self.architecture == "dilated_conv":
            return self.model(tokens)  # B x L

        # RiNALMo path: when LM is frozen, run it in no_grad to save memory
        if not self.finetune_lm:
            with torch.no_grad():
                representation = self.rinalmo(tokens)["representation"]  # B x L x E
            representation = representation.detach()
        else:
            representation = self.rinalmo(tokens)["representation"]  # B x L x E

        logits = self.pred_head(representation)  # B x L
        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        tokens, labels, atg_mask, target_mask = batch
        logits = self(tokens)

        loss = self.loss_fn(logits, labels, atg_mask, target_mask)

        self.log("train/loss", loss, sync_dist=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation / Test (shared logic)
    # ------------------------------------------------------------------
    def _eval_step(self, batch, log_prefix: str):
        tokens, labels, atg_mask, target_mask = batch
        logits = self(tokens)

        loss = self.loss_fn(logits, labels, atg_mask, target_mask)
        metrics = tis_binary_metrics(logits, labels, atg_mask, target_mask)

        # Collect per-position scores & labels (on CPU) for epoch-level top-k
        scores = torch.sigmoid(logits[target_mask]).detach().cpu()
        labs = labels[target_mask].detach().cpu()

        self.val_step_outputs.append((metrics, scores, labs))
        self.log(f"{log_prefix}/loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def _on_epoch_end(self, log_prefix: str):
        # Accumulate counts across batches
        accumulated = {}
        all_scores = []
        all_labels = []
        for metrics, scores, labs in self.val_step_outputs:
            for k, v in metrics.items():
                accumulated[k] = accumulated.get(k, torch.tensor(0, device=v.device)) + v
            all_scores.append(scores)
            all_labels.append(labs)

        agg = aggregate_tis_metrics(accumulated)

        # Top-k precision (computed on CPU from collected scores)
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        topk = tis_topk_precision(all_scores, all_labels)
        agg.update(topk)

        log = {f"{log_prefix}/{k}": v for k, v in agg.items()}
        self.log_dict(log, sync_dist=True, add_dataloader_idx=False)

        self.val_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, log_prefix="val")

    def on_validation_epoch_end(self):
        self._on_epoch_end(log_prefix="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, log_prefix="test")

    def on_test_epoch_end(self):
        self._on_epoch_end(log_prefix="test")

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.architecture == "dilated_conv":
            param_groups = [{"params": self.model.parameters()}]
        else:
            param_groups = [{"params": self.pred_head.parameters()}]
            if self.finetune_lm:
                param_groups.append(
                    {"params": self.rinalmo.transformer.parameters(), "lr": self.lr * 0.1}
                )

        optimizer = AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}


# ======================================================================
# CLI entry-point
# ======================================================================
def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Model -----------------------------------------------------------
    model = TISPredictionWrapper(
        architecture=args.architecture,
        lm_config=args.lm_config,
        head_embed_dim=args.head_embed_dim,
        head_num_blocks=args.head_num_blocks,
        head_kernel_size=args.head_kernel_size,
        head_dropout=args.head_dropout,
        finetune_lm=args.finetune_lm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        atg_lambda=args.atg_lambda,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        conv_dilations=args.conv_dilations,
        conv_dropout=args.conv_dropout,
        num_transformer_layers=args.num_transformer_layers,
        transformer_heads=args.transformer_heads,
    )

    if args.pretrained_rinalmo_weights:
        model.load_pretrained_rinalmo_weights(args.pretrained_rinalmo_weights)

    if args.init_params:
        model.load_state_dict(torch.load(args.init_params))

    # ---- DataModule ------------------------------------------------------
    alphabet = Alphabet()
    datamodule = TISDataModule(
        data_root=args.data_dir,
        test_data_root=args.test_data_dir,
        alphabet=alphabet,
        max_seq_len=args.max_seq_len,
        target_block_size=args.target_block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ---- Callbacks & loggers ---------------------------------------------
    callbacks = []
    loggers = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.wandb_experiment_name,
            save_dir=args.output_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_code=True,
        )
        loggers.append(wandb_logger)

    if args.log_lr and loggers:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if args.checkpoint_every_epoch:
        epoch_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename="tis_pred-epoch_ckpt-{epoch}-{step}",
            every_n_epochs=1,
            save_top_k=-1,
        )
        callbacks.append(epoch_ckpt_callback)

    if args.checkpoint_every_hour:
        time_ckpt_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename="tis_pred-latest-hourly-{epoch}-{step}",
            train_time_interval=timedelta(hours=1.0),
            save_top_k=1,
        )
        callbacks.append(time_ckpt_callback)

    # ---- Trainer ---------------------------------------------------------
    strategy = "auto"
    if args.devices != "auto" and (
        "," in args.devices
        or (int(args.devices) > 1 and int(args.devices) != -1)
    ):
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_every_n_steps,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
    )

    if args.data_dir:
        trainer.fit(model=model, datamodule=datamodule)
    if args.test_data_dir:
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Model ------------------------------------------------------------
    parser.add_argument(
        "--lm_config", type=str, default="giga",
        help="Language model configuration (giga, mega, micro, nano)",
    )
    parser.add_argument(
        "--pretrained_rinalmo_weights", type=str, default=None,
        help="Path to pretrained RiNALMo model weights",
    )
    parser.add_argument(
        "--init_params", type=str, default=None,
        help="Path to .pt file with full wrapper weights to resume from",
    )

    # --- Architecture selection --------------------------------------------
    parser.add_argument(
        "--architecture", type=str, default="rinalmo",
        choices=["rinalmo", "dilated_conv"],
        help="Model architecture: 'rinalmo' (pretrained LM + head) or 'dilated_conv' (standalone SpliceAI-style)",
    )

    # --- Prediction head (rinalmo architecture) ---------------------------
    parser.add_argument("--head_embed_dim", type=int, default=PRED_HEAD_EMBED_DIM)
    parser.add_argument("--head_num_blocks", type=int, default=PRED_HEAD_NUM_BLOCKS)
    parser.add_argument("--head_kernel_size", type=int, default=PRED_HEAD_KERNEL_SIZE)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument(
        "--finetune_lm", action="store_true", default=False,
        help="Unfreeze the pretrained RiNALMo LM and fine-tune it (default: frozen, head-only training)",
    )

    # --- Dilated-conv architecture hyper-parameters -----------------------
    parser.add_argument("--conv_channels", type=int, default=256)
    parser.add_argument("--conv_kernel_size", type=int, default=9)
    parser.add_argument("--conv_dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--conv_dropout", type=float, default=0.1)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)

    # --- Loss hyper-parameters (CompoundTISLoss: Tversky + ATG-contrastive) -
    parser.add_argument(
        "--tversky_alpha", type=float, default=0.3,
        help="False-positive weight in Tversky denominator (lower → tolerate more FP)",
    )
    parser.add_argument(
        "--tversky_beta", type=float, default=0.7,
        help="False-negative weight in Tversky denominator (higher → recall bias)",
    )
    parser.add_argument(
        "--atg_lambda", type=float, default=1.0,
        help="Mixing coefficient for the ATG-contrastive BCE term",
    )

    # --- Window geometry ---------------------------------------------------
    parser.add_argument(
        "--target_block_size", type=int, default=400,
        help="Size of the central target block (nt) where loss is computed. "
             "Flanking context fills the remainder of max_seq_len.",
    )

    # --- Data -------------------------------------------------------------
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory with train.csv and val.csv",
    )
    parser.add_argument(
        "--test_data_dir", type=str, default=None,
        help="Directory with test.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for checkpoints, logs, and temporary files",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max_seq_len", type=int, default=1022,
        help="Max nucleotide length per window (excl. CLS/EOS). 0 = no windowing.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)

    # --- Training ---------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--log_lr", action="store_true", default=False)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # --- Checkpointing ----------------------------------------------------
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False)
    parser.add_argument("--checkpoint_every_hour", action="store_true", default=False)

    # --- W&B --------------------------------------------------------------
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_experiment_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()
    main(args)
