
import argparse
from src.constants import *
import os
import subprocess
import datetime
from src.limo import LIMO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='selfies')
    parser.add_argument('--model_type', type=str, default='vae')
    parser.add_argument('--start_stage', type=int, default=0)
    parser.add_argument('--end_stage', type=int, default=10)
    parser.add_argument("-n", "--exp_name", type=str, default="default")
    parser.add_argument("-p", "--opt_method", type=str, default="default")
    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage
    model_type = args.model_type
    exp_suffix = args.config 
    exp_name:str = args.exp_name
    opt_method = args.opt_method

    print(f"Running limo using {exp_suffix} model {model_type}, starting from stage {start_stage}")
    
    tokenizer = args.config
    token_file = TOKENIZER_CONFIGS[args.config]["token_file"]
    log_file = f"temp/log_file_{exp_suffix}_{model_type}_{exp_name}.txt"
    # open(log_file, "w")

    limo = LIMO(token_file=token_file, tokenizer=tokenizer, model_type=model_type, exp_name=exp_name)
    
    # train VAE
    if start_stage <= 0:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}: Train {model_type}", flush=True, file=open(log_file, "a+"))
        limo.train_vae()

    # train property predictors
    if start_stage <= 1 and end_stage > 1:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Training PP", flush=True, file=open(log_file, "a+"))
        for prop_vals in [("ba", 10000)]:
            limo.train_property_predictor(
                prop=prop_vals[0], 
                num_mols=prop_vals[1])
        if (model_type == "vae") or (model_type == "vae_t"):
            for prop_vals in [("sa", 50000), ("qed", 50000)]:
                limo.train_property_predictor(
                    prop=prop_vals[0], 
                    num_mols=prop_vals[1])
        
    # generate molecules
    if start_stage <= 2 and end_stage > 2:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Generating molecules with opt method {opt_method}", flush=True, file=open(log_file, "a+"))
        # if (model_type == "vae") or (model_type == "vae_t"):
        #     opt_prop = "moba"
        #     sa_cut_off = 11
        #     qed_cut_off = -1
        # else:
        #     opt_prop = "ba"
        #     sa_cut_off = 11
        #     qed_cut_off = -1
        opt_prop = "moba"
        sa_cut_off = 11
        qed_cut_off = -1
        limo.generate_molecules(opt_prop=opt_prop, sa_cutoff=sa_cut_off, qed_cutoff=qed_cut_off, opt_method=opt_method)


    if start_stage <= 3 and end_stage > 3:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{model_type} Generating random molecules", flush=True, file=open(log_file, "a+"))
        limo.generate_random_molecules(num_mols=5000)


if __name__ == '__main__':
    main()