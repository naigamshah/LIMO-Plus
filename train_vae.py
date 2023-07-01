from utils import *
from models import *
import torch
import pytorch_lightning as pl
import argparse
from dataloaders import MolDataModule
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    args = parser.parse_args()
    exp_suffix = args.tokenizer #"gs_zinc"
    print(f"Training using {exp_suffix}")
    
    tokenizer = choose_tokenizer(args.tokenizer)
    dm = MolDataModule(1024, args.token_file, tokenizer)
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
            latent_dim=1024, embedding_dim=64)

    trainer = pl.Trainer(
        accelerator="cpu", 
        gpus=1, 
        max_epochs=18, 
        enable_checkpointing=False,
        logger=pl.loggers.CSVLogger('logs'),
        callbacks=[
            pl.callbacks.TQDMProgressBar(refresh_rate=1),
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
    torch.save(vae.state_dict(), f'vae_{exp_suffix}.pt')

if __name__ == '__main__':
    main()