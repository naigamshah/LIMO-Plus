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
from rdkit.Chem import MolFromSmiles, QED, CanonSmiles
from src.sascorer import calculateScore
from src.tokenizers import BaseTokenizer
from src.utils import *
import numpy as np
import json
from src.constants import *
# import multiprocessing
# try:
#     cpus = multiprocessing.cpu_count()
# except NotImplementedError:
#     cpus = 2   # arbitrary default

NUM_WORKERS = 4
class MolDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file, tokenizer, conditional=False):
        super(MolDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TokenizedDataset(file, tokenizer, conditional)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.8)), int(round(len(self.dataset) * 0.2))])
        print(f"DM len {len(self.dataset)}, Train size: {len(self.train_data)}, val size: {len(self.test_data)}")
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
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=False)
        

class TokenizedDataset(Dataset):
    def __init__(self, file, tokenizer: BaseTokenizer, conditional=False):
        self.tokenizer = tokenizer
        selfies = [line.split()[0] for line in open(file, 'r')]
        # pool = multiprocessing.Pool(processes=cpus)
        # parallelized = pool.map(self.tokenizer.encoder, smiles)
        # selfies = [x for x in tqdm(parallelized) if x]
        # selfies = [self.tokenizer.encoder(smile) for smile in tqdm(smiles)]
        self.alphabet = set()
        for s in selfies:
            self.alphabet.update(sf.split_selfies(s))
        self.alphabet = ['[pad]'] + ['[nop]'] + list(sorted(self.alphabet))
        self.max_len = max(len(list(sf.split_selfies(s)))+1 for s in selfies)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
        self.encodings = [([self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)] + [self.symbol_to_idx['[nop]']]) for s in selfies]
        self.conditional = conditional
        if conditional:
            prop_file = f'data/properties/{os.path.basename(file).replace(".txt", ".json")}'
            if os.path.exists(prop_file):
                props = json.load(open(prop_file, "r"))
                self.props = [props[s] for s in selfies]
            else:
                os.makedirs("data/properties/", exist_ok=True)
                props = json.load(open("data/zinc250k.json","r"))
                prop_dict = {tokenizer.encoder(k):v for k,v in tqdm(props.items(), desc="generating prop dict")}
                json.dump(prop_dict, open(prop_file, "w+"))
                self.props = [prop_dict[s] for s in selfies]
        print(f"Alphabet len is {len(self.alphabet)}, max len is {self.max_len}")
        print(f"Avg encoding len {np.mean([len(e) for e in self.encodings])}")

    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, i):
        item = {
            "x": torch.tensor(self.encodings[i] + [self.symbol_to_idx['[pad]'] for _ in range(self.max_len - len(self.encodings[i]))]),
        }
        if self.conditional:
            item["sa"] = (torch.tensor([self.props[i]["sa"]]) - SA_MEAN) / SA_STD
            item["qed"]= (torch.tensor([self.props[i]["qed"]]) - QED_MEAN) / QED_STD
        return item
    
    def smiles_to_indices(self, smiles):
        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(self.tokenizer.encoder(smiles))]
        return torch.tensor(encoding + [self.symbol_to_idx['[nop]']] + [self.symbol_to_idx['[pad]'] for i in range(self.max_len - len(encoding))])

    def smiles_to_one_hot(self, smiles):
        out = torch.zeros((self.max_len, len(self.symbol_to_idx)))
        for i, index in enumerate(self.smiles_to_indices(smiles)):
            out[i][index] = 1
        return out.flatten()
   
    def one_hot_to_selfies(self, hot):
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot.view((self.max_len, -1)).argmax(1)]).split("[nop]")[0].replace(' ', '')

    def one_hot_to_smiles(self, hot):
        return self.tokenizer.decoder(self.one_hot_to_selfies(hot))
    
    