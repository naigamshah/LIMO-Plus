import loss_landscapes
import torch 
from src.models import VAEFormer, PropertyPredictor
from loss_landscapes.metrics import Metric
from src.limo import LIMO
from src.constants import *
from src.utils import *
import numpy as np
import pickle
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--type', type=str, default="surrogate")
    parser.add_argument('--start_idx', type=int, default=-1)
    args = parser.parse_args()

    # metric = DecodeLoss()
    loss_type = args.type
    seed = args.seed
    model_type = "vae"
    exp_name = "default"
    tokenizer = "selfies"
    token_file = TOKENIZER_CONFIGS["selfies"]["token_file"]
    limo = LIMO(token_file=token_file, tokenizer=tokenizer, model_type=model_type, exp_name=exp_name)

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
    dm, model = limo.get_dm_model(True)
    model.eval()

    weights = {'ba': 5, 'sa': 2, 'qed': -8}
    props = {'logp': smiles_to_logp, 
            'penalized_logp': smiles_to_penalized_logp, 
            'qed': smiles_to_qed, 
            'sa': smiles_to_sa,
            'ba': lambda x: smiles_to_affinity(x, limo.autodock_executable, limo.protein_file, num_devices=num_devices)}
            
    def get_real_loss(x, weights):
        loss_d = {}
        with torch.no_grad():
            for idx,prop_name in enumerate(weights):
                smx = [dm.dataset.one_hot_to_smiles(x[i]) for i in range(x.shape[0])]
                prop_val = props[prop_name](smx)
                loss_d[prop_name]= np.array(prop_val) #* weights[prop_name]
        return loss_d

    def get_property_loss(x, weights):
        models = []
        #loss = 0 
        loss_d = {}
        with torch.no_grad():
            for idx,prop_name in enumerate(weights):
                #print(idx, prop_name)
                models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx)))
                models[-1].load_state_dict(torch.load(f'{PROP_MODELS_SAVE}/{prop_name}_{limo.save_model_suffix}.pt', map_location="cpu"))
                models[-1] = models[-1].to(device)
                prop_val = models[-1](x.to(model.device)).cpu()
                #loss += prop_val * weights[prop_name]
                loss_d[prop_name] = prop_val #* weights[prop_name]
        return loss_d
    
    def zToPredloss(z):
        with torch.no_grad():
            x = torch.exp(model.decode(z))
            #print(z)
            if loss_type == "real":
                l = get_real_loss(x, weights)
            else:
                l = get_property_loss(x, weights)
            return l

    # class DecodeLoss(Metric):
    #     """ Computes a specified loss function over specified input-output pairs. """
    #     def __init__(self):
    #         super().__init__()
    #     def __call__(self, dummyModel) -> float:
    #         return zToPredloss(dummyModel.modules[0].z.unsqueeze(0))
        
    # class dummyModel(torch.nn.Module):
    #     def __init__(self, *args, **kwargs) -> None:
    #         super().__init__(*args, **kwargs)
    #         self.z = torch.nn.Parameter(torch.zeros(model.latent_dim))
        
    
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

    zs = pickle.load(open("temp/optimized_z.pkl", "rb"))
    losses = zToPredloss(zs)
    pickle.dump(losses, open(f"temp/landscape_{loss_type}_{limo.save_model_suffix}_optimized_real_losses.pkl", "wb"))
    return

    if args.start_idx == -1:
        start = torch.zeros(model.latent_dim)
    else:
        start = pickle.load(open("temp/optimized_z.pkl", "rb"))[args.start_idx] #torch.zeros(model.latent_dim)

    print("start_norm", start.norm())
    d1 = torch.randn(model.latent_dim)
    d1 /= d1.norm()

    d2 = torch.randn(model.latent_dim)
    d2 = d2 - torch.dot(d2, d1) * d1 / (d1.norm()**2)
    d2 /= d2.norm()

    point_list = []
    distance = 40
    num_steps = 25
    step = 2 * distance / num_steps
    pos_list = []
    for i in range(-num_steps//2, num_steps//2):
        for j in range(-num_steps//2, num_steps//2):
            point = start + i * step * d1 + j * step * d2
            point_list.append(point)
            pos_list.append([(i,j)])

    losses = zToPredloss(torch.stack(point_list))
    landscape = {"point_list": point_list, "pos_list": pos_list, "loss": losses}

    # landscape = loss_landscapes.random_plane(dummyModel(), metric, normalization=None, distance=5, steps=100)

    pickle.dump(landscape, open(f"temp/landscape_{loss_type}_{limo.save_model_suffix}_{seed}_{args.start_idx}.pkl", "wb"))

if __name__ == '__main__':
    main()