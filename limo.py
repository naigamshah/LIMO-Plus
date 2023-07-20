
import argparse
from src.constants import *
import os
import subprocess

from src.train import *
from src.generate import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='selfies')
    parser.add_argument('--model_type', type=str, default='vae')
    parser.add_argument('--start_stage', type=int, default=0)
    parser.add_argument('--end_stage', type=int, default=10)
    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage
    model_type = args.model_type
    exp_suffix = args.config 
    print(f"Running limo using {exp_suffix} model {model_type}, starting from stage {start_stage}")
    
    tokenizer = args.config
    token_file = TOKENIZER_CONFIGS[args.config]["token_file"]
    log_file = f"temp/log_file_{exp_suffix}.txt"
    open(log_file, "w")
    
    # train VAE
    if start_stage <= 0:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}: Train {model_type}", flush=True, file=open(log_file, "a+"))
        train_vae(token_file=token_file, tokenizer=tokenizer, model_type=model_type)
        # result = subprocess.run(["python", "src/train/train_vae.py", f"--tokenizer {tokenizer} --token_file {token_file}"], check=True)
        # os.system(f"python src/train/train_vae.py --tokenizer {tokenizer} --token_file {token_file}")

    # train property predictors

    if start_stage <= 1 and end_stage > 1:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Training PP", flush=True, file=open(log_file, "a+"))
        for prop_vals in [("binding_affinity", 10000)]:
            train_property_predictor(
                token_file=token_file, 
                tokenizer=tokenizer,
                model_type=model_type,
                prop=prop_vals[0], 
                num_mols=prop_vals[1],
                autodock_executable=AUTODOCK_LOCATION)
        if model_type == "vae":
            for prop_vals in [("sa", 50000), ("qed", 50000)]:
                train_property_predictor(
                    token_file=token_file, 
                    tokenizer=tokenizer,
                    model_type=model_type,
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
    if start_stage <= 2 and end_stage > 2:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Generating molecules", flush=True, file=open(log_file, "a+"))
        if model_type == "vae":
            opt_prop = "multi_objective_binding_affinity"
            sa_cut_off = 5.5
            qed_cut_off = 0.4
        else:
            opt_prop = "binding_affinity"
            sa_cut_off = 11
            qed_cut_off = -1
        generate_molecules(token_file=token_file, tokenizer=tokenizer, model_type=model_type, opt_prop=opt_prop, sa_cutoff=sa_cut_off, qed_cutoff=qed_cut_off)
        # result = subprocess.run(["python", "src/generate/generate_molecules.py", f"--tokenizer {tokenizer} --token_file {token_file}"], check=True)
        # os.system(f"python src/generate/generate_molecules.py --tokenizer {tokenizer} --token_file {token_file}")
        # 

    if start_stage <= 3 and end_stage > 3:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Generating random molecules", flush=True, file=open(log_file, "a+"))
        generate_random_molecules(token_file=token_file, tokenizer=tokenizer, model_type=model_type, num_mols=5000)


if __name__ == '__main__':
    main()