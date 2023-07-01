import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import subprocess
import pickle
import selfies as sf
import csv
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, QED
from sascorer import calculateScore
from tokenizers import BaseTokenizer
from utils import *
# import multiprocessing
# try:
#     cpus = multiprocessing.cpu_count()
# except NotImplementedError:
#     cpus = 2   # arbitrary default

NUM_WORKERS = 4
class MolDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file, tokenizer):
        super(MolDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TokenizedDataset(file, tokenizer)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.8)), int(round(len(self.dataset) * 0.2))])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    
class PropDataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size):
        super(PropDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TensorDataset(x, y)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.9)), int(round(len(self.dataset) * 0.1))])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True)
        

class TokenizedDataset(Dataset):
    def __init__(self, file, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        selfies = [line.split()[0] for line in open(file, 'r')]
        # pool = multiprocessing.Pool(processes=cpus)
        # parallelized = pool.map(self.tokenizer.encoder, smiles)
        # selfies = [x for x in tqdm(parallelized) if x]
        # selfies = [self.tokenizer.encoder(smile) for smile in tqdm(smiles)]
        self.alphabet = set()
        for s in selfies:
            self.alphabet.update(sf.split_selfies(s))
        self.alphabet = ['[nop]'] + list(sorted(self.alphabet))
        self.max_len = max(len(list(sf.split_selfies(s))) for s in selfies)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
        self.encodings = [[self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)] for s in selfies]
        
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, i):
        return torch.tensor(self.encodings[i] + [self.symbol_to_idx['[nop]'] for _ in range(self.max_len - len(self.encodings[i]))])
    
    def smiles_to_indices(self, smiles):
        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(self.tokenizer.encoder(smiles))]
        return torch.tensor(encoding + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(encoding))])

    def smiles_to_one_hot(self, smiles):
        out = torch.zeros((self.max_len, len(self.symbol_to_idx)))
        for i, index in enumerate(self.smiles_to_indices(smiles)):
            out[i][index] = 1
        return out.flatten()
   
    def one_hot_to_selfies(self, hot):
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot.view((self.max_len, -1)).argmax(1)]).replace(' ', '')

    def one_hot_to_smiles(self, hot):
        return self.tokenizer.decoder(self.one_hot_to_selfies(hot))
    
    def one_hots_to_filter(self, hots):
        f = open('filter/to_filter.csv', 'w')
        for i, hot in enumerate(hots):
            f.write(f'{self.one_hot_to_smiles(hot)} {i}\n')
        f.close()
        subprocess.run('rd_filters filter --in filter/to_filter.csv --prefix filter/out', shell=True, stderr=subprocess.DEVNULL)
        out = []
        for row in csv.reader(open('filter/out.csv', 'r')):
            if row[0] != 'SMILES':
                out.append(int(row[2] == 'OK'))
        return out

    def one_hots_to_logp(self, hots):
        logps = []
        for i, hot in enumerate(hots):
            smile = self.one_hot_to_smiles(hot)
            try:
                logps.append(MolLogP(MolFromSmiles(smile)))
            except:
                logps.append(0)
        return logps


    def one_hots_to_qed(self, hots):
        qeds = []
        for i, hot in enumerate(tqdm(hots, desc='calculating QED')):
            smile = self.one_hot_to_smiles(hot)
            mol = MolFromSmiles(smile)
            qeds.append(QED.qed(mol))
        return qeds


    def one_hots_to_sa(self, hots):
        sas = []
        for i, hot in enumerate(tqdm(hots, desc='calculating SA')):
            smile = self.one_hot_to_smiles(hot)
            mol = MolFromSmiles(smile)
            sas.append(calculateScore(mol))
        return sas


    def one_hots_to_cycles(self, hots):
        cycles = []
        for hot in tqdm(hots, desc='counting undesired cycles'):
            smile = self.one_hot_to_smiles(hot)
            mol = MolFromSmiles(smile)
            cycle_count = 0
            for ring in mol.GetRingInfo().AtomRings():
                if not (4 < len(ring) < 7):
                    cycle_count += 1
            cycles.append(cycle_count)
        return cycles


    def one_hots_to_penalized_logp(self, hots):
        logps = []
        for i, hot in enumerate(hots):
            smile = self.one_hot_to_smiles(hot)
            mol = MolFromSmiles(smile)
            penalized_logp = MolLogP(mol) - calculateScore(mol)
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) > 6:
                    penalized_logp -= 1
            logps.append(penalized_logp)
        return logps

    def one_hots_to_affinity(self, hots, autodock, protein_file, num_devices=torch.cuda.device_count()):
        return smiles_to_affinity([self.one_hot_to_smiles(hot) for hot in hots], autodock, protein_file, num_devices=num_devices)
