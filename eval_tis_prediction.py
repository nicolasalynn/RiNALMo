"""Evaluate a trained TIS prediction checkpoint on the test set."""

import argparse
import torch
import pytorch_lightning as pl

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.tis_prediction.datamodule import TISDataModule
from train_tis_prediction import TISPredictionWrapper


def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    # Load model from Lightning checkpoint (hparams restored automatically)
    model = TISPredictionWrapper.load_from_checkpoint(
        args.checkpoint, map_location="cpu",
    )
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Architecture: {model.architecture}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # DataModule
    alphabet = Alphabet()
    datamodule = TISDataModule(
        data_root=None,
        test_data_root=args.test_data_dir,
        alphabet=alphabet,
        max_seq_len=args.max_seq_len,
        target_block_size=args.target_block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        logger=False,
    )

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TIS prediction on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory with test.csv")
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--target_block_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
