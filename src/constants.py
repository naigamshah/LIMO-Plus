AUTODOCK_LOCATION="../AutoDock-GPU/bin/autodock_gpu_128wi"
TEMP_FOLDER="temp"
TOKENS_PATH="tokens" 

GEN_MODELS_SAVE="saved_models/generative_models"
PROP_MODELS_SAVE="saved_models/property_models"

TOKENIZER_CONFIGS={
    "selfies":{
        "token_file": f"{TOKENS_PATH}/zinc_sf_selfies.txt"
    },
    "gs_zinc":{
        "token_file": f"{TOKENS_PATH}/zinc_gs_selfies.txt"
    },
    "gs_um":{
        "token_file": f"{TOKENS_PATH}/zinc_umgs_selfies.txt"
    }
}