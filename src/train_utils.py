from src.tokenizers import *
from src.dataloaders import *
from src.models import *

def get_dm_model(tokenizer, token_file, model_type="vae", load_from_ckpt=False):
    exp_suffix = tokenizer
    tokenizer_model = choose_tokenizer(tokenizer)
    token_loc = token_file

    if model_type == "cvae":
        modelClass = cVAE
        conditional = True
        latent_dim=1024
        embedding_dim=64
        batch_size = 1024
    elif model_type == "cvae_t":
        modelClass = cVAEFormer
        conditional = True
        latent_dim=1024
        embedding_dim=128
        batch_size = 256
    else: 
        modelClass = VAE
        conditional = False
        latent_dim=1024
        embedding_dim=64
        batch_size = 1024

    dm = MolDataModule(batch_size, token_loc, tokenizer_model, conditional=conditional)
    model = modelClass(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
        latent_dim=latent_dim, embedding_dim=embedding_dim)
    
    if load_from_ckpt:
        model.load_state_dict(torch.load(f'{GEN_MODELS_SAVE}/{model_type}/{model_type}_{exp_suffix}.pt'))
    return dm, model