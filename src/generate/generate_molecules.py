from src.utils import *
from src.models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from src.dataloaders import MolDataModule, PropDataModule
from src.constants import *
from src.tokenizers import *
import datetime

def generate_molecules(
    token_file,
    tokenizer,
    opt_prop='multi_objective_binding_affinity',
    num_mols=10000,
    top_k=3,
    sa_cutoff=5.5,
    qed_cutoff=0.4,
    optim_steps=10,
    autodock_executable='../AutoDock-GPU/bin/autodock_gpu_128wi',
    protein_file='data/1err/1err.maps.fld'
):
    
    exp_suffix = tokenizer #"gs_zinc"
    print(f"Generating using {exp_suffix}")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}: Generating molecules", flush=True, file=open(f"temp/log_file_{exp_suffix}.txt", "a+"))
    
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
    vae.load_state_dict(torch.load(f'{GEN_MODELS_SAVE}/vae_{exp_suffix}.pt', map_location="cpu"))
    vae.eval()
    

    def get_optimized_z(weights, num_mols, num_steps=10):
        models = []
        for prop_name in weights:
            models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx)))
            models[-1].load_state_dict(torch.load(f'{PROP_MODELS_SAVE}/{prop_name}_{exp_suffix}.pt', map_location="cpu"))
            models[-1] = models[-1].to(device)
        z = torch.randn((num_mols, 1024), device=device, requires_grad=True)
        optimizer = optim.Adam([z], lr=0.1)
        losses = []
        for epoch in tqdm.tqdm(range(num_steps), desc='generating molecules'):
            optimizer.zero_grad()
            loss = 0
            probs = torch.exp(vae.decode(z))
            for i, model in enumerate(models):
                out = model(probs)
                loss += torch.sum(out) * list(weights.values())[i]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return z


    def get_prop(prop, x):
        smx = [dm.dataset.one_hot_to_smiles(x[i]) for i in range(x.shape[0])]
        return torch.tensor(props[prop](smx), device=device).unsqueeze(1).float()

    def run_multiobjective_and_filtering(num_mols, max_sa, min_qed):
        weights = {'binding_affinity': 5, 'sa': 2, 'qed': -8}
        #weights = {'sa': 2, 'qed': -8}
        z = get_optimized_z(weights, num_mols)
        with torch.no_grad():
            x = torch.exp(vae.decode(z))
        cycles = get_prop('cycles', x)
        x = x[cycles.flatten() == 0]
        sa = get_prop('sa', x)
        x = x[sa.flatten() < max_sa]
        sa = sa[sa.flatten() < max_sa]
        qed = get_prop('qed', x)
        x = x[qed.flatten() > min_qed]
        qed = qed[qed.flatten() > min_qed]
        binding_affinity = get_prop('binding_affinity', x)
        kd = np.exp(binding_affinity.detach().cpu().numpy() / (0.00198720425864083 * 298.15)).flatten()
        sa = sa.detach().cpu().numpy()
        qed = qed.detach().cpu().numpy()
        return [dm.dataset.one_hot_to_smiles(hot) for hot in x], kd, sa, qed


    props = {'logp': smiles_to_logp, 
            'penalized_logp': smiles_to_penalized_logp, 
            'qed': smiles_to_qed, 
            'sa': smiles_to_sa,
            'binding_affinity': lambda x: smiles_to_affinity(x, autodock_executable, protein_file, num_devices=num_devices),
            'cycles': smiles_to_cycles}

    if opt_prop == 'multi_objective_binding_affinity':
        smiles, prop, _, _ = run_multiobjective_and_filtering(num_mols, sa_cutoff, qed_cutoff)
    else:
        z = get_optimized_z({opt_prop: (1 if opt_prop in ('sa', 'binding_affinity') else -1)}, num_mols, num_steps=optim_steps)
        with torch.no_grad():
            x = torch.exp(vae.decode(z))
        smiles = [dm.dataset.one_hot_to_smiles(hot) for hot in x]
        prop = get_prop(opt_prop, x).detach().cpu().numpy().flatten()
        
    if opt_prop in ('sa', 'binding_affinity', 'multi_objective_binding_affinity'):
        if not os.path.exists(f"gen_mols/"):
            os.makedirs(f"gen_mols/")
        pickle.dump([(prop[i], smiles[i]) for i in range(len(smiles))], 
                  open(f"gen_mols/{exp_suffix}.pkl", "wb"))
        for i in np.argpartition(prop, top_k)[:top_k]:
            if opt_prop == 'binding_affinity':
                print(delta_g_to_kd(prop[i]), smiles[i])
            else:
                print(prop[i], smiles[i])
    else:
        for i in np.argpartition(prop, -top_k)[-top_k:]:
            print(prop[i], smiles[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_file', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity', 'multi_objective_binding_affinity'], default='multi_objective_binding_affinity')
    parser.add_argument('--num_mols', type=int, default=10000)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--sa_cutoff', type=float, default=5.5)
    parser.add_argument('--qed_cutoff', type=float, default=0.4)
    parser.add_argument('--optim_steps', type=int, default=10)
    parser.add_argument('--autodock_executable', type=str, default='../AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
    args = parser.parse_args()

    generate_molecules(
        args.token_file,
        args.tokenizer,
        args.prop,
        args.num_mols,
        args.top_k,
        args.sa_cutoff,
        args.qed_cutoff,
        args.optim_steps,
        args.autodock_executable,
        args.protein_file
    )