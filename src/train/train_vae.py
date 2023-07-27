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

def train_vae(token_file, tokenizer, model_type="vae"):
    exp_suffix =  tokenizer
    print(f"Train {model_type} using {exp_suffix}")
    
    dm, model = get_dm_model(tokenizer=tokenizer, token_file=token_file, model_type=model_type)

    trainer = pl.Trainer(
        accelerator="gpu", 
        num_nodes=1,
        #max_epochs=18, 
        max_steps=30000,
        enable_checkpointing=False,
        logger=pl.loggers.CSVLogger(f'temp/logs/{model_type}_{exp_suffix}'),
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
    if not os.path.exists(f"{GEN_MODELS_SAVE}/{model_type}"):
        os.makedirs(f"{GEN_MODELS_SAVE}/{model_type}")
    torch.save(model.state_dict(), f'{GEN_MODELS_SAVE}/{model_type}/{model_type}_{exp_suffix}.pt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    args = parser.parse_args()

    train_vae(args.token_file, args.tokenizer)