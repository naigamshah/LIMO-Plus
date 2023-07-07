
import argparse
from src.constants import *
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='selfies')
    args = parser.parse_args()


    exp_suffix = args.config 
    print(f"Training using {exp_suffix}")
    
    tokenizer = CONFIGS[args.config]["tokenizer"]
    token_file = CONFIGS[args.config]["token_file"]
    
    # train VAE
    os.system(f"python src/train/train_vae.py --tokenizer {tokenizer} --token_file {token_file}")

    # train property predictors
    for prop_vals in [("sa", 100000), ("qed", 100000), ("ba", 10000)]:
        os.system(f"python src/train/train_property_predictors.py --tokenizer {tokenizer} --token_file {token_file} \
                  --prop {prop_vals[0]} --num_mols {prop_vals[1]} --autodock_executable {AUTODOCK_LOCATION} ")
        
    # generate molecules
    os.system(f"python src/generate/generate_molecules.py --tokenizer {tokenizer} --token_file {token_file}")    


if __name__ == '__main__':
    main()