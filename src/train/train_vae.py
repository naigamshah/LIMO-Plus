import torch
import pytorch_lightning as pl
import argparse
from src.utils import *
from src.models import *
from src.dataloaders import MolDataModule
from src.constants import *
from src.tokenizers import *

def train_vae(token_file, tokenizer):
    exp_suffix =  tokenizer
    print(f"Train VAE using {exp_suffix}")
    
    tokenizer = choose_tokenizer(tokenizer)
    token_loc = token_file

    dm = MolDataModule(1024, token_loc, tokenizer)
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
            latent_dim=1024, embedding_dim=64)

    trainer = pl.Trainer(
        accelerator="gpu", 
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
    trainer.fit(vae, dm)
    print('Saving..')
    if not os.path.exists(f"{GEN_MODELS_SAVE}"):
        os.makedirs(f"{GEN_MODELS_SAVE}")
    torch.save(vae.state_dict(), f'{GEN_MODELS_SAVE}/vae_{exp_suffix}.pt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    args = parser.parse_args()

    train_vae(args.token_file, args.tokenizer)