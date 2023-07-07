AUTODOCK_LOCATION="../AutoDock-GPU/bin/autodock_gpu_128wi"
TOKENS_PATH="tokens/" 

GEN_MODELS_SAVE="saved_models/generative_models"
PROP_MODELS_SAVE="saved_models/property_models"

CONFIGS={
    "selfies":{
        "token_file": "zinc250k.smi",
        "tokenizer": "tokens/zinc_sf_seflies.txt"
    },
    "gs_zinc":{
        "token_file": "zinc_gs_selfies.txt",
        "tokenizer": "tokens/zinc_gs_seflies.txt"
    }
}