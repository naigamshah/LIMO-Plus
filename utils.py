
import torch
import subprocess
import pickle
import csv
from tqdm import tqdm
import os
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, QED
from sascorer import calculateScore
import time
import math
from models import VAE
from tokenizers import BaseTokenizer

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))

def smiles_to_z(smiles, vae: VAE, dataset):
    device = vae.device
    zs = torch.zeros((len(smiles), 1024), device=device)
    for i, smile in enumerate(tqdm(smiles)):
        target = dataset.smiles_to_one_hot(smile).to(device)
        z = vae.encode(dataset.smiles_to_indices(smile).unsqueeze(0).to(device))[0].detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=0.1)
        for epoch in range(10000):
            optimizer.zero_grad()
            loss = torch.mean((torch.exp(vae.decode(z)[0]) - target) ** 2)
            loss.backward()
            optimizer.step()
        zs[i] = z.detach()
    return zs
    

def smiles_to_penalized_logp(smiles):
    logps = []
    for i, smile in enumerate(smiles):
        mol = MolFromSmiles(smile)
        penalized_logp = MolLogP(mol) - calculateScore(mol)
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 6:
                penalized_logp -= 1
        logps.append(penalized_logp)
    return logps

def smiles_to_affinity(smiles, autodock, protein_file, num_devices=torch.cuda.device_count()):
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(num_devices):
        os.mkdir(f'ligands/{device}')
    device = 0
    for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(f'obabel -:"{smiles[i]}" -O ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == num_devices:
            device = 0
    while True:
        total = 0
        for device in range(num_devices):
            total += len(os.listdir(f'ligands/{device}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -s 0 -L ligands/0/ligand0.pdbqt -N outs/ligand0', shell=True, stdout=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -s 0 -B ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL))
        stop = False
        while not stop: 
            for p in ps:
                stop = True
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir('outs'), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'outs/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(subprocess.check_output(f"grep 'RANKING' outs/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1", shell=True).decode('utf-8').strip())
    return [min(affin, 0) for affin in affins]

def choose_tokenizer(tokenizer_type: str) -> BaseTokenizer:
    if tokenizer_type == "selfies":
        from tokenizers import SelfiesTokenizer
        return SelfiesTokenizer()
    elif "gs" in tokenizer_type:
        from tokenizers import GroupSelfiesTokenizer
        style = tokenizer_type.split("_")[1]
        if style == "zinc":
            return GroupSelfiesTokenizer("tokens/zinc_gs_grammar.txt")
        elif style == "templates":
            return GroupSelfiesTokenizer("tokens/templates_grammar.txt")
        else:
            raise Exception(f"wrong style expected zinc or templates, got {style}")

    raise Exception(f"No tokenizer found for type {tokenizer_type}")     

# if os.path.exists('dm.pkl'):
#     dm = pickle.load(open('dm.pkl', 'rb'))