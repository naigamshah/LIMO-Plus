import os
from utils import *
import argparse
from dataloaders import MolDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, default='zinc250k.smi')
    parser.add_argument('--tokenizer', type=str, default='selfies')
    args = parser.parse_args()
    print('Preprocessing..')
    tokenizer = choose_tokenizer(args.tokenizer)
    dm = MolDataModule(1024, args.tokens, tokenizer)

    dm_path = f"dm_{args.tokenizer}.pkl"
    pickle.dump(dm, open(dm_path, 'wb'))
    print('Done!')