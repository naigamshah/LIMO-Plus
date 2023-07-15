from src.tokenizers import *
from src.dataloaders import *
from src.models import *

def get_dm_model(tokenizer, token_file, load_from_ckpt=False):
    exp_suffix = tokenizer
    tokenizer_model = choose_tokenizer(tokenizer)
    token_loc = token_file
    conditional=True
    dm = MolDataModule(1024, token_loc, tokenizer_model, conditional=conditional)
    modelClass = cVAE if conditional else VAE
    model = modelClass(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
        latent_dim=1024, embedding_dim=64)
    
    if load_from_ckpt:
        model_prefix = "cvae" if conditional else "vae" 
        model.load_state_dict(torch.load(f'{GEN_MODELS_SAVE}/{model_prefix}_{exp_suffix}.pt'))
    return dm, model