from src.utils import *
from src.models import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
import argparse
import time
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
replacements = ('N', 'O', 'Cl', 'F')

def finetune(
    smiles,
    autodock_executable='../AutoDock-GPU/bin/autodock_gpu_128wi',
    protein_file='data/1err/1err.maps.fld'
):
    base = smiles
    while True:
        tests = [base]
        for i in range(len(base)):
            if base[i] == 'C':
                for replace in replacements:
                    test = list(base)
                    test[i] = replace
                    test = ''.join(test)
                    m = MolFromSmiles(test)
                    if m:
                        for other in replacements:
                            if m.HasSubstructMatch(MolFromSmiles(f'{replace}{other}')):
                                break
                            if MolFromSmiles(f'{replace}={other}') and m.HasSubstructMatch(MolFromSmiles(f'{replace}={other}')):
                                break
                        else:
                            tests.append(test)
        tests *= 10
        affins = smiles_to_affinity(tests, autodock_executable, protein_file, 1)
        base = tests[np.argmin(affins)]
        print(delta_g_to_kd(min(affins)), base)
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str)
    parser.add_argument('--autodock_executable', type=str, default='../AutoDock-GPU/bin/autodock_gpu_128wi')
    parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
    args = parser.parse_args()

    finetune(
        args.smiles,
        args.autodock_executable,
        args.protein_file
    )