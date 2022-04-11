import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from train.globals import DATASET_ARGS
from train.mt5_finetuner import MT5Finetuner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(42)
    os.environ["WANDB_API_KEY"] = '1a8b66b615fabe2e3d956af808171179815bcc09'
    wandb_logger = WandbLogger(project='spanish_gec')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=DATASET_ARGS.default_root_dir, monitor="val_loss", mode="min", save_top_k=1
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.05, patience=3, verbose=False, mode='min'
    )

    train_params = dict(
        default_root_dir=DATASET_ARGS.default_root_dir,
        accelerator='gpu',
        devices=1,
        max_epochs=15,
        learning_rate=1e-3,
        accumulate_grad_batches=16,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger,
        callbacks=[early_stopping_callback]
    )

    model = MT5Finetuner(DATASET_ARGS)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
