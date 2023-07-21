AUTODOCK_LOCATION="../AutoDock-GPU/bin/autodock_gpu_64wi"
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
    "gs_czinc":{
        "token_file": f"{TOKENS_PATH}/zinc_gsc_selfies.txt"
    },
    "gs_um":{
        "token_file": f"{TOKENS_PATH}/zinc_umgs_selfies.txt"
    },
    "gs_use":{
        "token_file": f"{TOKENS_PATH}/zinc_usegs_selfies.txt"
    }
}

MAX_MOLS_CHUNK = 10000

SA_SCALING = 10
QED_SCALING = 1
SA_MEAN, SA_STD = 3.05, 0.83
QED_MEAN, QED_STD = 0.73, 0.14
SA_TARGET = 2
QED_TARGET= 0.8
