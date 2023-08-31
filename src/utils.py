
import torch
import subprocess
import pickle
import csv
from tqdm import tqdm
import os
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, QED
from src.sascorer import calculateScore
import time
import math
from src.models import VAE
from rdkit import RDLogger
import tempfile

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))


def smiles_to_filter(smiles):
        f = open('filter/to_filter.csv', 'w')
        for i, smile in enumerate(smiles):
            f.write(f'{smile} {i}\n')
        f.close()
        subprocess.run('rd_filters filter --in filter/to_filter.csv --prefix filter/out', shell=True, stderr=subprocess.DEVNULL)
        out = []
        for row in csv.reader(open('filter/out.csv', 'r')):
            if row[0] != 'SMILES':
                out.append(int(row[2] == 'OK'))
        return out

def smiles_to_logp(smiles):
    logps = []
    for i, smile in enumerate(smiles):
        try:
            logps.append(MolLogP(MolFromSmiles(smile)))
        except:
            logps.append(0)
    return logps


def smiles_to_qed(smiles):
    qeds = []
    for i, smile in enumerate(tqdm(smiles, desc='calculating QED')):
        mol = MolFromSmiles(smile)
        qeds.append(QED.qed(mol))
    return qeds


def smiles_to_sa(smiles):
    sas = []
    for i, smile in enumerate(tqdm(smiles, desc='calculating SA')):
        mol = MolFromSmiles(smile)
        sas.append(calculateScore(mol))
    return sas


def smiles_to_cycles(smiles):
    cycles = []
    for smile in tqdm(smiles, desc='counting undesired cycles'):
        mol = MolFromSmiles(smile)
        cycle_count = 0
        for ring in mol.GetRingInfo().AtomRings():
            if not (4 < len(ring) < 7):
                cycle_count += 1
        cycles.append(cycle_count)
    return cycles


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

def smiles_to_affinity(smiles, autodock, protein_file, num_devices=torch.cuda.device_count()):
    with tempfile.TemporaryDirectory(dir="/dev/shm/", prefix="temp_") as temp_folder:
        if not os.path.exists(f'{temp_folder}/ligands'):
            os.makedirs(f'{temp_folder}/ligands')
        if not os.path.exists(f'{temp_folder}/outs'):
            os.makedirs(f'{temp_folder}/outs')
        subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
        subprocess.run(f'rm {temp_folder}/outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
        subprocess.run(f'rm {temp_folder}/outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
        subprocess.run(f'rm -rf {temp_folder}/ligands/*', shell=True, stderr=subprocess.DEVNULL)
        for device in range(num_devices):
            os.mkdir(f'{temp_folder}/ligands/{device}')
        device = 0
        for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
            subprocess.Popen(f'obabel -:"{smiles[i]}" -O {temp_folder}/ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
            device += 1
            if device == num_devices:
                device = 0
        while True:
            total = 0
            for device in range(num_devices):
                total += len(os.listdir(f'{temp_folder}/ligands/{device}'))
            if total == len(smiles):
                break
        time.sleep(1)
        print('running autodock..', flush=True)
        if len(smiles) == 1:
            subprocess.run(f'{autodock} -M {protein_file} -s 0 -L {temp_folder}/ligands/0/ligand0.pdbqt -N {temp_folder}/outs/ligand0', shell=True, stdout=subprocess.DEVNULL)
        else:
            ps = []
            for device in range(num_devices):
                ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -s 0 -B {temp_folder}/ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1} -n 5', shell=True, stdout=subprocess.DEVNULL))
            stop = False
            while not stop: 
                for p in ps:
                    stop = True
                    if p.poll() is None:
                        time.sleep(1)
                        stop = False
        affins = [0 for _ in range(len(smiles))]
        for file in tqdm(os.listdir(f'{temp_folder}/outs'), desc='extracting binding values'):
            if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'{temp_folder}/outs/{file}').read():
                affins[int(file.split('ligand')[1].split('.')[0])] = float(subprocess.check_output(f"grep 'RANKING' {temp_folder}/outs/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1", shell=True).decode('utf-8').strip())
        return [min(affin, 0) for affin in affins]


# if os.path.exists('dm.pkl'):
#     dm = pickle.load(open('dm.pkl', 'rb'))