# Spanish Grammatical Error Correction (GEC)

This repository contains code to train an mT5 model from HuggingFace on the COWS-L2H dataset.
A project writeup can be found [here](https://diligent-raver-536.notion.site/Spanish-Grammatical-Error-Correction-88d0f0d1d090412baf4c52cdf87a0468).

### Setup
It is recommended to set up this repository on a GPU for training, 8GB RAM is minumum to train a small mT5 model.
The conda environment can be created using:
```
>>> conda env create -f environment.yml
```
You can add this project to your PYTHONPATH using:
```
>>> export PYTHONPATH="</path/to/spanish_gec/>":$PYTHONPATH
```

### Data Preprocessing
The data is cleaned and already available in `cowsl2h/data`.
To clean the source text yourself, you can download the original dataset from the [COWS-L2H GitHub](https://github.com/ucdaviscl/cowsl2h)
and use the following script:
```
>>> python process_dataset <path/to/cowsl2h/csv>
```

### Training
You can start finetuning the model:
```
>>> export WANDB_API_KEY="<Your WandB API Key>"
>>> python run_train.py
```
The following files can be used to set various hyperparameters:
<br>`cowsl2h.py` -- dataset loading
<br>`globals.py` -- dataset parameters
<br>`mt5_finetuner` -- training loop, loss computation, WandB logging
<br>`run_train` -- training hyperparameters 

### Prediction
You can run inference on all datasets:
```
>>> python run_predict.py </path/to/ckpt/dir> </path/to/ckpt/file> -d <train, val, test, or all> -b <num_beams>
```

### Evaluation
You can compute the F-0.5 score using ERRANT:
```
>>> ./eval.sh <path/to/predicted/sentences/>
```
This will output a `results-cs.txt` in your predictions directory with the several metrics:
>TPs, FPs, FNs, Precision, Recall, F-0.5

You can also use the following Colab notebook to try individual inferences with a sample finetuned model.