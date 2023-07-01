from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
from dataloaders import MolDataModule, PropDataModule

def main():
    pl.seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity'], default='binding_affinity')
    parser.add_argument('--num_mols', type=int, default=10000)
    parser.add_argument('--autodock_executable', type=str, default='../AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
    args = parser.parse_args()
    exp_suffix = args.tokenizer #"gs_zinc"
    print(f"Training using {exp_suffix}")
    
    tokenizer = choose_tokenizer(args.tokenizer)
    dm = MolDataModule(1024, args.token_file, tokenizer)
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     num_devices = torch.cuda.device_count()
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps") #torch.device('cuda'  else 'cpu')
    #     num_devices = 1
    # else:
    #     device = torch.device("cpu")
    #     num_devices = 1

    device = torch.device("cpu")
    num_devices = 1

    try:
        vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
    except NameError:
        raise Exception('No dm.pkl found, please run preprocess_data.py first')
    vae.load_state_dict(torch.load(f'vae_{exp_suffix}.pt'))
    vae.eval()


    # def generate_training_mols(num_mols, prop_func):
    #     xs = []
    #     ys = []
    #     batch_size = 1024
    #     num_chunks = (num_mols // batch_size) + 1
    #     mols_done = 0
    #     with torch.no_grad():
    #         for i in tqdm(range(num_chunks), desc="generating molecules in chunks"): 
    #             nmols = min(batch_size, num_mols - mols_done)
    #             z = torch.randn((nmols, 1024), device="cpu")
    #             x = torch.exp(vae.to("cpu").decode(z)).cpu()
    #             y = torch.tensor(prop_func(x), device="cpu").unsqueeze(1).float()
    #             xs.append(x)
    #             ys.append(y)
    #             mols_done += nmols
    #     return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        #return x, y
    def generate_training_mols(num_mols, prop_func):
        with torch.no_grad():
            z = torch.randn((num_mols, 1024), device=device)
            x = torch.exp(vae.decode(z))
            pickle.dump(x, open(f"property_models/{args.prop}_{exp_suffix}_x", 'wb')) 
            y = torch.tensor(prop_func(x), device=device).unsqueeze(1).float()
            pickle.dump(y, open(f"property_models/{args.prop}_{exp_suffix}_y", 'wb')) 
        return x, y

    props = {'logp': dm.dataset.one_hots_to_logp, 
            'penalized_logp': dm.dataset.one_hots_to_penalized_logp, 
            'qed': dm.dataset.one_hots_to_qed, 
            'sa': dm.dataset.one_hots_to_sa,
            'binding_affinity': lambda x: dm.dataset.one_hots_to_affinity(x, args.autodock_executable, args.protein_file, num_devices=num_devices)}
    print("Generating training mols")
    x, y = generate_training_mols(args.num_mols, props[args.prop])
    print("Done!")
    model = PropertyPredictor(x.shape[1])
    dm = PropDataModule(x[1000:], y[1000:], 100)
    trainer = pl.Trainer(
        accelerator="gpu", 
        gpus=1, 
        max_epochs=5,  
        enable_checkpointing=False, 
        auto_lr_find=True,
        logger=pl.loggers.CSVLogger('logs'),
        callbacks=[
            pl.callbacks.TQDMProgressBar(refresh_rate=500),
        ])
    trainer.tune(model, dm)
    trainer.fit(model, dm)
    model.eval()
    model = model.to(device)

    print(f'property predictor trained, correlation of r = {linregress(model(x[:1000].to(device)).detach().cpu().numpy().flatten(), y[:1000].detach().cpu().numpy().flatten()).rvalue}')
    if not os.path.exists('property_models'):
        os.mkdir('property_models')
    torch.save(model.state_dict(), f'property_models/{args.prop}_{exp_suffix}.pt')

if __name__ == '__main__':
    main()