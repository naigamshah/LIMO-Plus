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
    def __init__(self, grammar_file) -> None:
        super().__init__()
        self.grammar = GroupGrammar.from_file(grammar_file) # | GroupGrammar.essential_set()

    def encoder(self, x):
        mol = Chem.MolFromSmiles(x)
        extracted = self.grammar.extract_groups(mol)
        return self.grammar.encoder(mol, extracted)
    
    def decoder(self, x):
        mol = self.grammar.decoder(x)
        return Chem.MolToSmiles(mol)
    
def extract_groups_from_zinc():
    zinc = [x.strip() for x in open("zinc250k.smi")]
    fragments = fragment_mols(zinc, convert=True, method='default')
    vocab_fragment = dict([(f'frag{idx}', Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments)])
    grammar = GroupGrammar(vocab=vocab_fragment)
    grammar.to_file('tokens/zinc_gs_grammar.txt')

def gs_tokenize_zinc():
    tokenizer = GroupSelfiesTokenizer("tokens/zinc_gs_grammar.txt")
    smiles = [line.split()[0] for line in open("zinc250k.smi", 'r')]
    # pool = multiprocessing.Pool(processes=8)
    # parallelized = pool.map(tokenizer.encoder, smiles)
    # selfies = [x for x in tqdm(parallelized) if x]
    selfies = [tokenizer.encoder(smile) for smile in tqdm.tqdm(smiles)]
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

if __name__ == "__main__":
    # extract_groups_from_zinc()
    gs_tokenize_zinc()
    #sf_tokenize_zinc()