# Experiments with LIMO


## Installation

Please ensure that [RDKit](https://www.rdkit.org/docs/Install.html) and [Open Babel](https://openbabel.org/wiki/Category:Installation) are installed. The following Python packages are also required (these can also be installed with `pip install -r requirements.txt`):

```
torch
pytorch-lightning==1.9.0
selfies
scipy
tqdm
```

Code was tested with Python 3.9, but will likely work on any version of Python 3.

## Tokenizer config

Use `src/tokenizers.py` script to generate data using different tokenizers.

## Training + generation

Use the `run_limo.py` script with different arguments to run different stages (training, fine-tuning, generation), different tokenizer configurations, different optimization strategies etc.

