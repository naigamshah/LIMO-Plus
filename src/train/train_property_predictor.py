import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
from src.utils import *
from src.models import *
from src.dataloaders import MolDataModule, PropDataModule
from src.constants import *
from src.tokenizers import *

def train_property_predictor(
    token_file,
    tokenizer,
    prop,
    num_mols=10000,
    autodock_executable='../AutoDock-GPU/bin/autodock_gpu_128wi',
    protein_file='data/1err/1err.maps.fld'
):
    exp_suffix = tokenizer
    print(f"Train PP using {exp_suffix}")
    temp_folder = TEMP_FOLDER + f"_{exp_suffix}"
    
    tokenizer = choose_tokenizer(tokenizer)
    dm = MolDataModule(1024, token_file, tokenizer)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
        #device = torch.device("mps") #torch.device('cuda'  else 'cpu')
        num_devices = 1
    else:
        device = torch.device("cpu")
        num_devices = 1

    try:
        vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
    except NameError:
        raise Exception('No dm.pkl found, please run preprocess_data.py first')
    vae.load_state_dict(torch.load(f'{GEN_MODELS_SAVE}/vae_{exp_suffix}.pt'))
    vae.eval()

    def generate_training_mols(num_mols, prop_func):
        # if os.path.exists(f"property_models/{args.prop}_{exp_suffix}_y"):
        #     print("loading existing")
        #     return pickle.load(open(f"property_models/{args.prop}_{exp_suffix}_x","rb")), pickle.load(open(f"property_models/{args.prop}_{exp_suffix}_y", "rb"))
        with torch.no_grad():
            xs = []
            num_chunks = num_mols // MAX_MOLS_CHUNK 
            if num_mols % MAX_MOLS_CHUNK != 0:
                num_chunks += 1
            idx = range(0, num_mols)
            for i in range(num_chunks):
                z = torch.randn((len(idx[i*MAX_MOLS_CHUNK:(i+1)*MAX_MOLS_CHUNK]), 1024), device=device)
                xs.append(torch.exp(vae.decode(z)))

            x = torch.cat(xs, dim=0)
            print(f"{exp_suffix}:{prop}: Decoding variance = {x.std()}", flush=True, file=open(f"temp/log_file_{exp_suffix}.txt", "a+"))
            # pickle.dump(x, open(f"property_models/{args.prop}_{exp_suffix}_x", 'wb')) 
            smx = [dm.dataset.one_hot_to_smiles(x[i]) for i in range(x.shape[0])]
            y = torch.tensor(prop_func(smx), device=device).unsqueeze(1).float()
            # pickle.dump(y, open(f"property_models/{args.prop}_{exp_suffix}_y", 'wb')) 
        return x, y

    props = {'logp': smiles_to_logp, 
            'penalized_logp': smiles_to_penalized_logp, 
            'qed': smiles_to_qed, 
            'sa': smiles_to_sa,
            'binding_affinity': lambda x: smiles_to_affinity(x, autodock_executable, protein_file, num_devices=num_devices)}
    
    print("Generating training mols")
    x, y = generate_training_mols(num_mols, props[prop])
    print("Done!")
    model = PropertyPredictor(x.shape[1])
    dm = PropDataModule(x[1000:], y[1000:], 1000)
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=1,
        max_epochs=5,  
        enable_checkpointing=False, 
        logger=pl.loggers.CSVLogger('temp/logs'),
        callbacks=[
            pl.callbacks.TQDMProgressBar(refresh_rate=500),
        ])
    trainer.fit(model, dm)
    model.eval()
    model = model.to(device)

    print(f'{exp_suffix}:{prop}: property predictor trained, correlation of r = {linregress(model(x[:1000].to(device)).detach().cpu().numpy().flatten(), y[:1000].detach().cpu().numpy().flatten()).rvalue}', flush=True, file=open(f"temp/log_file_{exp_suffix}.txt", "a+"))
    if not os.path.exists(f'{PROP_MODELS_SAVE}'):
        os.makedirs(f'{PROP_MODELS_SAVE}')
    torch.save(model.state_dict(), f'{PROP_MODELS_SAVE}/{prop}_{exp_suffix}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity'], default='binding_affinity')
    parser.add_argument('--num_mols', type=int, default=10000)
    parser.add_argument('--autodock_executable', type=str, default='../AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='data/1err/1err.maps.fld')
    args = parser.parse_args()

    train_property_predictor(
        args.token_file,
        args.tokenizer,
        args.prop,
        args.num_mols,
        args.autodock_executable,
        args.protein_file,
    )