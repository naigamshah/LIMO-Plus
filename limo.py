
import argparse
from src.constants import *
import os
import subprocess

from src.train import *
from src.generate import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='selfies')
    args = parser.parse_args()


    exp_suffix = args.config 
    print(f"Training using {exp_suffix}")
    
    tokenizer = args.config
    token_file = TOKENIZER_CONFIGS[args.config]["token_file"]
    
    # train VAE
    train_vae(token_file=token_file, tokenizer=tokenizer)
    # result = subprocess.run(["python", "src/train/train_vae.py", f"--tokenizer {tokenizer} --token_file {token_file}"], check=True)
    # os.system(f"python src/train/train_vae.py --tokenizer {tokenizer} --token_file {token_file}")

    # train property predictors
    for prop_vals in [("sa", 100000), ("qed", 100000), ("binding_affinity", 10000)]:
        train_property_predictor(
            token_file=token_file, 
            tokenizer=tokenizer,
            prop=prop_vals[0], 
            num_mols=prop_vals[1],
            autodock_executable=AUTODOCK_LOCATION)
        # os.system(f"python src/train/train_property_predictors.py --tokenizer {tokenizer} --token_file {token_file} \
        #           --prop {prop_vals[0]} --num_mols {prop_vals[1]} --autodock_executable {AUTODOCK_LOCATION} ")
        # result = subprocess.run(["python", "src/train/train_property_predictors.py", 
        #                          f"--tokenizer {tokenizer} --token_file {token_file} \
        #                          --prop {prop_vals[0]} --num_mols {prop_vals[1]} --autodock_executable {AUTODOCK_LOCATION}"], 
        #                          check=True)
    
        
    # generate molecules
    generate_molecules(token_file=token_file, tokenizer=tokenizer)
    # result = subprocess.run(["python", "src/generate/generate_molecules.py", f"--tokenizer {tokenizer} --token_file {token_file}"], check=True)
    # os.system(f"python src/generate/generate_molecules.py --tokenizer {tokenizer} --token_file {token_file}")    


if __name__ == '__main__':
    main()