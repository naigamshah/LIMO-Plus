import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
from src.utils import *
from src.models import *
from src.dataloaders import MolDataModule, PropDataModule
from src.constants import *
from src.tokenizers import *
import datetime
from src.pcgrad import PCGrad
from src.optimizers.opt_methods import *

class LIMO:
    def __init__(self, token_file, tokenizer, model_type="vae", exp_name="default") -> None:
        self.token_file = token_file
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.exp_name = exp_name

        self.autodock_executable='../AutoDock-GPU/bin/autodock_gpu_64wi'
        self.protein_file='data/1err/1err.maps.fld'
        self.save_logs_suffix = f'{self.model_type}/{self.tokenizer}/{self.exp_name}'
        self.save_model_suffix = f'{self.model_type}_{self.tokenizer}_{self.exp_name}'

    def get_dm_model(self, load_from_ckpt=False):
        exp_suffix = self.tokenizer
        tokenizer_model = choose_tokenizer(self.tokenizer)
        token_loc = self.token_file

        if self.model_type == "cvae":
            modelClass = cVAE
            load_prop_list = None
            latent_dim=1024
            embedding_dim=64
            batch_size = 1024
            prop_dim = latent_dim
            autoreg = False
            use_z_surrogate = False
        elif self.model_type == "cvae_t":
            modelClass = VAEFormer
            load_prop_list = None
            latent_dim=128
            embedding_dim=128
            batch_size = 256
            prop_dim = latent_dim
            autoreg = False
            use_z_surrogate = False
        elif self.model_type == "vae_t":
            modelClass = VAEFormer
            load_prop_list = ["ba", "sa", "qed"]
            latent_dim=128
            embedding_dim=128
            batch_size = 256
            prop_dim = latent_dim
            autoreg = True
            use_z_surrogate = True
        else: 
            modelClass = VAE
            load_prop_list = None
            latent_dim=1024
            embedding_dim=64
            batch_size = 1024
            prop_dim = latent_dim
            autoreg = False
            use_z_surrogate = False

        dm = MolDataModule(batch_size, token_loc, tokenizer_model, load_prop_list=load_prop_list, wpad=autoreg)
        
        model = modelClass(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
            latent_dim=latent_dim, embedding_dim=embedding_dim, autoreg=autoreg, use_z_surrogate=use_z_surrogate)
        
        if load_from_ckpt:
            model.load_state_dict(torch.load(f'{GEN_MODELS_SAVE}/{self.save_model_suffix}.pt'))
        return {"dm": dm, "gen_model": model}

    def train_vae(self):
        exp_suffix =  self.tokenizer
        print(f"Train {self.model_type} using {exp_suffix}")
        
        dm_model = self.get_dm_model()
        dm, model = dm_model["dm"], dm_model["gen_model"]

        trainer = pl.Trainer(
            accelerator="gpu", 
            num_nodes=1,
            #max_epochs=18, 
            max_steps=100000,
            enable_checkpointing=False,
            logger=pl.loggers.CSVLogger(f'temp/logs/{self.save_logs_suffix}'),
            callbacks=[
                pl.callbacks.TQDMProgressBar(refresh_rate=500)
            ])
        print('Training..')
        trainer.fit(model, dm)
        if not os.path.exists(f"{GEN_MODELS_SAVE}"):
            os.makedirs(f"{GEN_MODELS_SAVE}")
        if trainer.state.status == "finished":
            print('Saving..')
            torch.save(model.state_dict(), f'{GEN_MODELS_SAVE}/{self.save_model_suffix}.pt')

    def train_property_predictor(
        self,
        prop,
        num_mols=10000
    ):
        exp_suffix = self.tokenizer
        
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

        
        dm_model = self.get_dm_model(load_from_ckpt=True)
        dm, gen_model = dm_model["dm"], dm_model["gen_model"]
        gen_model.to(device)
        gen_model.eval()

        if gen_model.use_z_surrogate:
            return

        def generate_training_mols(num_mols, prop_func):
            # if os.path.exists(f"property_models/{args.prop}_{exp_suffix}_y"):
            #     print("loading existing")
            #     return pickle.load(open(f"property_models/{args.prop}_{exp_suffix}_x","rb")), pickle.load(open(f"property_models/{args.prop}_{exp_suffix}_y", "rb"))
            with torch.no_grad():
                xs = []
                num_chunks = num_mols // MAX_MOLS_NOGRAD_CHUNK 
                if num_mols % MAX_MOLS_NOGRAD_CHUNK != 0:
                    num_chunks += 1
                idx = range(0, num_mols)
                for i in range(num_chunks):
                    z = torch.randn((len(idx[i*MAX_MOLS_NOGRAD_CHUNK:(i+1)*MAX_MOLS_NOGRAD_CHUNK]), gen_model.latent_dim), device=device)
                    xs.append(torch.exp(gen_model.decode(z)))

                x = torch.cat(xs, dim=0)
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{prop}: Decoding variance = {x.std()}", 
                      flush=True, 
                      file=open(f"temp/log_file_{self.tokenizer}_{self.model_type}_{self.exp_name}.txt", "a+"))
                # pickle.dump(x, open(f"property_models/{args.prop}_{exp_suffix}_x", 'wb')) 
                smx = [dm.dataset.one_hot_to_smiles(x[i]) for i in range(x.shape[0])]
                y = torch.tensor(prop_func(smx), device=device).unsqueeze(1).float()
                # pickle.dump(y, open(f"property_models/{args.prop}_{exp_suffix}_y", 'wb')) 
            return x, y

        props = {'logp': smiles_to_logp, 
                'penalized_logp': smiles_to_penalized_logp, 
                'qed': smiles_to_qed, 
                'sa': smiles_to_sa,
                'ba': lambda x: smiles_to_affinity(x, self.autodock_executable, self.protein_file, num_devices=num_devices)}
        
        print("Generating training mols")
        x, y = generate_training_mols(num_mols, props[prop])
        print("Done!")
        prop_model = PropertyPredictor(x.shape[1])
        dm = PropDataModule(x[1000:], y[1000:], 1000)
        trainer = pl.Trainer(
            accelerator="gpu",
            num_nodes=1,
            max_epochs=5,  
            enable_checkpointing=False, 
            logger=pl.loggers.CSVLogger(f'temp/logs/{self.save_logs_suffix}/prop'),
            callbacks=[
                pl.callbacks.TQDMProgressBar(refresh_rate=500),
            ])
        trainer.fit(prop_model, dm)

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:{exp_suffix}:{prop}: property predictor trained, correlation of r = {linregress(prop_model(x[:1000].to(device)).detach().cpu().numpy().flatten(), y[:1000].detach().cpu().numpy().flatten()).rvalue}', 
              flush=True, 
              file=open(f"temp/log_file_{self.tokenizer}_{self.model_type}_{self.exp_name}.txt", "a+"))
        if not os.path.exists(f'{PROP_MODELS_SAVE}'):
            os.makedirs(f'{PROP_MODELS_SAVE}')
        if trainer.state.status == "finished":
            print('Saving..')
            torch.save(prop_model.state_dict(), f'{PROP_MODELS_SAVE}/{prop}_{self.save_model_suffix}.pt')

    def generate_molecules(
        self,
        opt_prop='moba',
        num_mols=10000,
        top_k=3,
        sa_cutoff=5.5,
        qed_cutoff=0.4,
        optim_steps=100,
        opt_method="default"
    ):
        
        exp_suffix = self.tokenizer #"gs_zinc"
        print(f"Generating using {exp_suffix}")
        
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

        dm_model = self.get_dm_model(load_from_ckpt=True)
        dm, gen_model = dm_model["dm"], dm_model["gen_model"]
        gen_model.to(device)
        gen_model.eval()

        def get_optimized_z(weights, num_mols, num_steps=10):
            models = []

            if gen_model.use_z_surrogate:
                for idx,prop_name in enumerate(weights):
                    models.append(gen_model.z_surrogates[prop_name])
            else:
                for idx,prop_name in enumerate(weights):
                    #print(idx, prop_name)
                    models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx)))
                    models[-1].load_state_dict(torch.load(f'{PROP_MODELS_SAVE}/{prop_name}_{self.save_model_suffix}.pt', map_location="cpu"))
                    models[-1] = models[-1].to(device)
            
            task_specific_params = []
            for m in models:
                task_specific_params += m.parameters()
            # train_dm = dm.train_dataloader()
            # train_dm_iter = iter(train_dm)
            # MAX_MOLS_GRAD_CHUNK = train_dm.batch_size
            num_chunks = num_mols // MAX_MOLS_GRAD_CHUNK 
            if num_mols % MAX_MOLS_GRAD_CHUNK != 0:
                num_chunks += 1
            idx = range(0, num_mols)
            zs = []
            for i in range(num_chunks):
                z = torch.randn((len(idx[i*MAX_MOLS_GRAD_CHUNK:(i+1)*MAX_MOLS_GRAD_CHUNK]), gen_model.latent_dim), device=device, requires_grad=True)
                # with torch.no_grad():
                #     z_, _, _ = gen_model.encode(next(train_dm_iter)["x"].to(device))
                # z = z_.clone().detach().requires_grad_(True)
                if opt_method != "default":
                    weight_method = WeightMethods(
                        opt_method, n_tasks=3, device=device
                    )
                    optimizer = torch.optim.Adam(
                        [
                            dict(params=[z], lr=1),
                            dict(params=weight_method.parameters(), lr=1),
                        ],
                    )
                    # optimizer = PCGrad(optim.Adam([z], lr=0.1))
                else:
                    optimizer = optim.Adam([z], lr=1)
                losses = []
                for epoch in tqdm.tqdm(range(num_steps), desc='generating molecules'):
                    loss = 0
                    objectives = []
                    if gen_model.use_z_surrogate:
                        for modeli, model in enumerate(models):
                            out = model(z)
                            objectives.append(torch.sum(out) * list(weights.values())[modeli])
                            loss += torch.sum(out) * np.sign(list(weights.values())[modeli])
                    else:
                        probs = torch.exp(gen_model.decode(z))
                        for modeli, model in enumerate(models):
                            out = model(probs)
                            objectives.append(torch.sum(out) * list(weights.values())[modeli])
                            loss += torch.sum(out) * list(weights.values())[modeli]
                    optimizer.zero_grad()
                    if opt_method != "default":
                        stacked_objectives = torch.stack(
                            objectives
                        )
                        loss, extra_outputs = weight_method.backward(
                            losses=stacked_objectives,
                            shared_parameters=list(gen_model.decoder.parameters())+[z],
                            task_specific_parameters=list(task_specific_params),
                        )
                        #optimizer.pc_backward(objectives)
                    else:
                        loss.backward()
                    
                    optimizer.step()
                    losses.append(loss.item())
                zs.append(z.detach())
            return torch.cat(zs, dim=0)


        def get_prop(prop, x):
            smx = [dm.dataset.one_hot_to_smiles(x[i]) for i in range(x.shape[0])]
            return torch.tensor(props[prop](smx), device=device).unsqueeze(1).float()

        def run_multiobjective_and_filtering(num_mols, max_sa, min_qed):
            weights = {'ba': 5, 'sa': 2, 'qed': -8}
            #weights = {'sa': 2, 'qed': -8}
            z = get_optimized_z(weights, num_mols)

            # pickle.dump(z.cpu(), open("temp/optimized_z_train.pkl", "wb"))
            
            with torch.no_grad():
                x = torch.exp(gen_model.decode(z))
            # cycles = get_prop('cycles', x)
            # x = x[cycles.flatten() == 0]
            sa = get_prop('sa', x)
            x = x[sa.flatten() < max_sa]
            sa = sa[sa.flatten() < max_sa]
            qed = get_prop('qed', x)
            x = x[qed.flatten() > min_qed]
            qed = qed[qed.flatten() > min_qed]
            binding_affinity = get_prop('ba', x)
            kd = np.exp(binding_affinity.detach().cpu().numpy() / (0.00198720425864083 * 298.15)).flatten()
            sa = sa.detach().cpu().numpy()
            qed = qed.detach().cpu().numpy()
            return [dm.dataset.one_hot_to_smiles(hot) for hot in x], kd, sa, qed


        props = {'logp': smiles_to_logp, 
                'penalized_logp': smiles_to_penalized_logp, 
                'qed': smiles_to_qed, 
                'sa': smiles_to_sa,
                'ba': lambda x: smiles_to_affinity(x, self.autodock_executable, self.protein_file, num_devices=num_devices),
                'cycles': smiles_to_cycles}

        if opt_prop == 'moba':
            smiles, prop, _, _ = run_multiobjective_and_filtering(num_mols, sa_cutoff, qed_cutoff)
        else:
            z = get_optimized_z({opt_prop: (1 if opt_prop in ('sa', 'ba') else -1)}, num_mols, num_steps=optim_steps)
            with torch.no_grad():
                x = torch.exp(gen_model.decode(z))
            smiles = [dm.dataset.one_hot_to_smiles(hot) for hot in x]
            prop = get_prop(opt_prop, x).detach().cpu().numpy().flatten()
            
        if opt_prop in ('sa', 'ba', 'moba'):
            if not os.path.exists(f"gen_mols"):
                os.makedirs(f"gen_mols")
            if opt_method != "default":
                pickle.dump([(prop[i], smiles[i]) for i in range(len(smiles))], 
                    open(f"gen_mols/{opt_prop}_{opt_method}_{self.save_model_suffix}.pkl", "wb"))
            else:
                pickle.dump([(prop[i], smiles[i]) for i in range(len(smiles))], 
                    open(f"gen_mols/{opt_prop}_{self.save_model_suffix}.pkl", "wb"))
            for i in np.argpartition(prop, top_k)[:top_k]:
                if opt_prop == 'ba':
                    print(delta_g_to_kd(prop[i]), smiles[i])
                else:
                    print(prop[i], smiles[i])
        else:
            for i in np.argpartition(prop, -top_k)[-top_k:]:
                print(prop[i], smiles[i])

    
    def generate_random_molecules(
        self,
        num_mols=5000,
    ):
        exp_suffix = self.tokenizer #"gs_zinc"
        print(f"Generating random using {exp_suffix}")
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{exp_suffix}:{self.model_type}: Generating random molecules", flush=True, file=open(f"temp/log_file_{exp_suffix}.txt", "a+"))
        
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

        
        dm_model = self.get_dm_model(load_from_ckpt=True)
        dm, model = dm_model["dm"], dm_model["gen_model"]
        model.to(device)
        model.eval()


        z = torch.randn((num_mols, model.latent_dim), device=device, requires_grad=False)
        with torch.no_grad():
            x = torch.exp(model.decode(z))

        
        smiles = [dm.dataset.one_hot_to_smiles(hot) for hot in x]

        if not os.path.exists(f"gen_mols"):
            os.makedirs(f"gen_mols")

        pickle.dump([smiles[i] for i in range(len(smiles))], 
                    open(f"gen_mols/random_{self.save_model_suffix}.pkl", "wb"))
