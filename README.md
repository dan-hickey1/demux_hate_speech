This repository contains code adapted from the Demux-MEmo library (https://github.com/gchochla/Demux-MEmo/blob/master/README.md) adapted to work with the Measuring Hate Speech Corpus.

#### Instructions for use

To train a model, run:
`python3 experiments/demux.py MHS --model_name bert-base-uncased --root_dir {root_dir} --train_split train --dev_split dev --model_save --max_length 512 --early_stopping_patience 5 --correct_bias --num_train_epochs 20 --dropout_prob 0.1  --device cuda --reps {reps} --platform reddit`

Arguments:
- use `--model_save` if you want to save the model
- use `--platform reddit` if you want to train on Reddit data only, otherwise use `--platform all` (default is all)
- type `MHS` after the Python script name if you want to train using aggregated labels for each post, use `MHSAnnotators` if you want to train using individual annotator labels.
- to use basic BERT instead of Demux, run `python3 experiments/base_train.py` instead of `experiments/demux.py`
