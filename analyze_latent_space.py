import loss_landscapes
import torch 
from src.models import VAEFormer, PropertyPredictor
from src.limo import LIMO
from src.constants import *
from src.utils import *
import numpy as np
import pickle
import random
import argparse
from src.eval.dirichlet import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--type', type=str, default="surrogate")
    parser.add_argument('--start_idx', type=int, default=-1)
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-n", "--exp_name", type=str, default="default")
    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
        #device = torch.device("mps") #torch.device('cuda'  else 'cpu')
        num_devices = 1
    else:
        device = torch.device("cpu")
        num_devices = 1

    # metric = DecodeLoss()
    loss_type = args.type
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


    model_type = args.model_type #"vae"
    exp_name = args.exp_name #"zsurr"
    tokenizer = args.config #"selfies"
    token_file = TOKENIZER_CONFIGS["selfies"]["token_file"]
    limo = LIMO(token_file=token_file, tokenizer=tokenizer, model_type=model_type, exp_name=exp_name)


    # custom_config = dict(
    #     # modelClass = VAEFormer,
    #     # load_prop_list = ["ba", "sa", "qed"],
    #     # latent_dim=128,
    #     # embedding_dim=128,
    #     # batch_size = 256,
    #     # autoreg = True,
    #     # use_z_surrogate = True

    #     modelClass = VAE,
    #     load_prop_list = ["ba", "sa", "qed"],
    #     latent_dim=1024,
    #     embedding_dim=64,
    #     batch_size = 1024,
    #     autoreg = False,
    #     use_z_surrogate = True if "zsurr" in exp_name else False,
    # )
    dm_model = limo.get_dm_model(load_from_ckpt=True)
    dm, model = dm_model["dm"], dm_model["gen_model"]
    model.to(device)
    model.eval()

    weights = {'sa': 2, 'qed': -8, 'ba': 5}
    props = {'logp': smiles_to_logp, 
            'penalized_logp': smiles_to_penalized_logp, 
            'qed': smiles_to_qed, 
            'sa': smiles_to_sa,
            'ba': lambda x: smiles_to_affinity(x, limo.autodock_executable, limo.protein_file, num_devices=num_devices)}
            
    
    
    if os.path.exists(f"temp/latent_space_{limo.save_model_suffix}.pkl"):
        data_dict = pickle.load(open(f"temp/latent_space_{limo.save_model_suffix}.pkl", "rb"))
    else:
        data_dict = {"z": [], "sa": [], "qed": [], "ba": []}
        with torch.no_grad():
            for batch in iter(dm.train_dataloader()):
                #print(batch)
                x, sa, qed, ba = batch["x"], batch["sa"], batch["qed"], batch["ba"]
                z = model.encode(x.to(model.device))[0]
                for i in range(z.shape[0]):
                    data_dict["z"].append(z[i].cpu().detach().float().numpy()) 
                    data_dict["sa"].append(sa[i].cpu().detach().float().numpy()*SA_STD+SA_MEAN)
                    data_dict["qed"].append(qed[i].cpu().detach().float().numpy()*QED_STD+QED_MEAN)
                    data_dict["ba"].append(ba[i].cpu().detach().float().numpy()*BA_STD+BA_MEAN)
                # if len(data_dict["z"]) > 1000:
                #     break
            with open(f"temp/latent_space_{limo.save_model_suffix}.pkl", "wb") as f:
                pickle.dump(data_dict, f)
    
    z = np.array(data_dict["z"])
    sa = np.array(data_dict["sa"])
    qed = np.array(data_dict["qed"])
    ba = np.array(data_dict["ba"])

    #selected_pts = np.random.permutation(z.shape[0])[:10000]
    #print("Total smoothness")
    surr_val = 0
    with torch.no_grad():
        models = []
        z_inp = torch.tensor(z).to(model.device)
        if model.use_z_surrogate:
            #for idx,prop_name in enumerate(weights):
            prop_name = "sa"
            surr_val += np.sign(weights[prop_name]) * model.z_surrogates[prop_name](z_inp)#*SA_STD+SA_MEAN)
            prop_name = "qed"
            surr_val += np.sign(weights[prop_name]) * model.z_surrogates[prop_name](z_inp)#*QED_STD+QED_MEAN)
            prop_name = "ba"
            surr_val += np.sign(weights[prop_name]) * model.z_surrogates[prop_name](z_inp)#*BA_STD+BA_MEAN)
            real_val = (((sa-SA_MEAN)/SA_STD)*np.sign(weights["sa"])+((qed-QED_MEAN)/QED_STD)*np.sign(weights["qed"])+((ba-BA_MEAN)/BA_STD)*np.sign(weights["ba"]))
        else:
            probs = torch.exp(model.decode(z_inp))
            for idx,prop_name in enumerate(weights):
                #print(idx, prop_name)
                models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx)))
                models[-1].load_state_dict(torch.load(f'{PROP_MODELS_SAVE}/{prop_name}_{limo.save_model_suffix}.pt', map_location="cpu"))
                models[-1] = models[-1].to(device)
                surr_val += weights[prop_name] * models[-1](probs)
            real_val = (sa*weights["sa"]+qed*weights["qed"]+ba*weights["ba"])

    print()
    print("Total corr")
    #get_smoothnes_kNN_sparse(z[selected_pts], (sa*weights["sa"]+qed*weights["qed"]+ba*weights["ba"])[selected_pts])
    #engy_dist = get_dirichlet_energy(z[selected_pts], (sa*weights["sa"]+qed*weights["qed"]+ba*weights["ba"])[selected_pts])
    #engy_dist = get_dirichlet_energy_faiss(z, (sa*weights["sa"]+qed*weights["qed"]+ba*weights["ba"]))
    diff_dist = get_corr(z, real_val, surr_val)
    diff_dist["MSE"] = torch.nn.functional.mse_loss(torch.tensor(real_val), surr_val).item()

    with open(f"temp/corr_dist_{limo.save_model_suffix}.pkl", "wb") as f:
        pickle.dump(diff_dist, f)

if __name__ == '__main__':
    main()