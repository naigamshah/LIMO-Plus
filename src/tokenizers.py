import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from group_selfies import (
    fragment_mols, 
    Group, 
    MolecularGraph, 
    GroupGrammar, 
    group_encoder
)
import tqdm
import multiprocessing

class BaseTokenizer:
    def encoder(self, x:str):
        raise NotImplementedError
    def decoder(self):
        raise NotImplementedError

class SelfiesTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
    def encoder(self, x):
        return sf.encoder(x)
    def decoder(self, x):
        return sf.decoder(x)
    
class GroupSelfiesTokenizer(BaseTokenizer):
    def __init__(self, grammar_file, append_essential=False) -> None:
        super().__init__()
        self.grammar = GroupGrammar.from_file(grammar_file) 
        if append_essential:
            self.grammar = self.grammar | GroupGrammar.essential_set()

    def encoder(self, x):
        mol = Chem.MolFromSmiles(x)
        extracted = self.grammar.extract_groups(mol)
        return self.grammar.encoder(mol, extracted)
    
    def decoder(self, x):
        mol = self.grammar.decoder(x)
        return Chem.MolToSmiles(mol)
    
def extract_groups_from_zinc():
    zinc = [x.strip() for x in open("tokens/zinc250k.smi")]
    fragments = fragment_mols(zinc, convert=True, method='mmpa')
    vocab_fragment = dict([(f'frag{idx}', Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments)])
    grammar = GroupGrammar(vocab=vocab_fragment)
    grammar.to_file('tokens/zinc_gsm_grammar.txt')


def um_gs_tokenize_zinc():
    tokenizer = GroupSelfiesTokenizer("tokens/um_gs_grammar.txt")
    smiles = [line.split()[0] for line in open("tokens/zinc250k.smi", 'r')]
    # pool = multiprocessing.Pool(processes=8)
    # parallelized = pool.map(tokenizer.encoder, smiles)
    # selfies = [x for x in tqdm(parallelized) if x]
    selfies = [tokenizer.encoder(smile) for smile in tqdm.tqdm(smiles)]
    with open("tokens/zinc_umgs_selfies.txt", "w") as f:
        for selfie in selfies:
            f.write(f"{selfie}\n") 

def use_gs_tokenize_zinc():
    tokenizer = GroupSelfiesTokenizer("tokens/use_gs_grammar.txt", append_essential=True)
    smiles = [line.split()[0] for line in open("tokens/zinc250k.smi", 'r')]
    # pool = multiprocessing.Pool(processes=8)
    # parallelized = pool.map(tokenizer.encoder, smiles)
    # selfies = [x for x in tqdm(parallelized) if x]
    selfies = [tokenizer.encoder(smile) for smile in tqdm.tqdm(smiles)]
    with open("tokens/zinc_usegs_selfies.txt", "w") as f:
        for selfie in selfies:
            f.write(f"{selfie}\n") 

def gs_tokenize_zinc():
    tokenizer = GroupSelfiesTokenizer("tokens/zinc_gs_grammar.txt")
    smiles = [line.split()[0] for line in open("zinc250k.smi", 'r')]
    pool = multiprocessing.Pool(processes=8)
    parallelized = pool.map(lambda x: tokenizer.encoder(x), smiles)
    selfies = [x for x in tqdm(parallelized) if x]
    #selfies = [tokenizer.encoder(smile) for smile in tqdm.tqdm(smiles)]
    with open("tokens/zinc_gsc_selfies.txt", "w") as f:
        for selfie in selfies:
            f.write(f"{selfie}\n") 
    #selfies = [tokenizer.encoder(smile) for smile in tqdm(smiles)]

def sf_tokenize_zinc():
    tokenizer = SelfiesTokenizer()
    smiles = [line.split()[0] for line in open("zinc250k.smi", 'r')]
    # pool = multiprocessing.Pool(processes=8)
    # parallelized = pool.map(tokenizer.encoder, smiles)
    # selfies = [x for x in tqdm(parallelized) if x]
    selfies = [tokenizer.encoder(smile) for smile in tqdm.tqdm(smiles)]
    with open("tokens/zinc_sf_selfies.txt", "w") as f:
        for selfie in selfies:
            f.write(f"{selfie}\n") 

def choose_tokenizer(tokenizer_type: str) -> BaseTokenizer:
    if tokenizer_type == "selfies":
        from src.tokenizers import SelfiesTokenizer
        return SelfiesTokenizer()
    elif "gs" in tokenizer_type:
        from src.tokenizers import GroupSelfiesTokenizer
        style = tokenizer_type.split("_")[1]
        if style == "zinc":
            return GroupSelfiesTokenizer("tokens/zinc_gs_grammar.txt")
        elif style == "um":
            return GroupSelfiesTokenizer("tokens/um_gs_grammar.txt")
        elif style == "use":
            return GroupSelfiesTokenizer("tokens/use_gs_grammar.txt", append_essential=True)
        else:
            raise Exception(f"wrong style expected zinc or templates, got {style}")
        

    raise Exception(f"No tokenizer found for type {tokenizer_type}")     

if __name__ == "__main__":
    #extract_groups_from_zinc()
    #gs_tokenize_zinc()
    #sf_tokenize_zinc()
    #um_gs_tokenize_zinc()
    use_gs_tokenize_zinc()