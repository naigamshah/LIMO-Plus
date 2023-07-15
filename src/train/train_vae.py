import torch
import pytorch_lightning as pl
import argparse
from src.utils import *
from src.models import *
from src.dataloaders import MolDataModule
from src.constants import *
from src.tokenizers import *
from src.train_utils import *
import datetime

def train_vae(token_file, tokenizer):
    exp_suffix =  tokenizer
    print(f"Train VAE using {exp_suffix}")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}: Train VAE", flush=True, file=open(f"temp/log_file_{exp_suffix}.txt", "a+"))
    
    dm, model = get_dm_model(tokenizer=tokenizer, token_file=token_file)

    trainer = pl.Trainer(
        accelerator="cpu", 
        num_nodes=1,
        max_epochs=18, 
        enable_checkpointing=False,
        logger=pl.loggers.CSVLogger('temp/logs'),
        callbacks=[
            pl.callbacks.TQDMProgressBar(refresh_rate=500),
            # pl.callbacks.ModelCheckpoint(
            #     dirpath="./checkpoints",
            #     monitor="val_loss",
            #     save_weights_only=True,
            #     save_last=True,
            #     every_n_epochs=1,
            # )
        ])
    print('Training..')
    trainer.fit(model, dm)
    print('Saving..')
    if not os.path.exists(f"{GEN_MODELS_SAVE}"):
        os.makedirs(f"{GEN_MODELS_SAVE}")
    torch.save(model.state_dict(), f'{GEN_MODELS_SAVE}/cvae_{exp_suffix}.pt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    args = parser.parse_args()

    train_vae(args.token_file, args.tokenizer)