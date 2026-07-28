#!/bin/bash
# Train-on-whole models used for SHAP attribution and OOD inference.
#
# The --max_epochs value for each run is the mean early-stopping epoch
# averaged over all 8 CV folds from the corresponding run_experiments.sh run.
# Early stopping used a patience of 500 epochs on the validation loss.
cd "$(dirname "$0")"

COMMON_RNN="--database DATA/database.csv
  --batch_size 128 --seed 100
  --cv_folds 8 --folds 0
  --lr 0.001 --l2_lambda 0.0
  --hidden_size 100 --dropout_prob 0.3
  --energy_window_low -3.0 --energy_window_high 3.0
  --energy_window_interpolation_pts 200
  --keepE --split_at_fermi --use_layernorm --save_model
  --rnn_type bigru --lstm_layers 3
  --use_channel_self_attention --use_temporal_attention
  --channel_self_attn_dim 32
  --train_on_whole_data"

COMMON_CNN="--database DATA/database.csv
  --batch_size 128 --seed 100
  --cv_folds 8 --folds 0
  --lr 0.001 --l2_lambda 0.0
  --hidden_size 100 --dropout_prob 0.3
  --energy_window_low -3.0 --energy_window_high 3.0
  --energy_window_interpolation_pts 200
  --keepE --split_at_fermi --use_layernorm --save_model
  --rnn_type cnn --cnn_depth 4 --noise_height_fraction 0.0
  --train_on_whole_data"

python main.py $COMMON_RNN --max_epochs  944 --output experiments/mae              --target MAE              --experiment_name rnn_allorbitals_train_on_whole
python main.py $COMMON_RNN --max_epochs  949 --output experiments/mae              --target MAE              --keep_d_only --experiment_name rnn_dorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs 1008 --output experiments/mae              --target MAE              --experiment_name cnn_allorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs  765 --output experiments/mae              --target MAE              --keep_d_only --experiment_name cnn_dorbitals_train_on_whole

python main.py $COMMON_RNN --max_epochs 1248 --output experiments/mae+esoc         --target MAE+ESOC         --experiment_name rnn_allorbitals_train_on_whole
python main.py $COMMON_RNN --max_epochs 1112 --output experiments/mae+esoc         --target MAE+ESOC         --keep_d_only --experiment_name rnn_dorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs 1431 --output experiments/mae+esoc         --target MAE+ESOC         --experiment_name cnn_allorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs 1496 --output experiments/mae+esoc         --target MAE+ESOC         --keep_d_only --experiment_name cnn_dorbitals_train_on_whole

python main.py $COMMON_RNN --max_epochs 1150 --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --experiment_name rnn_allorbitals_train_on_whole
python main.py $COMMON_RNN --max_epochs 1188 --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --keep_d_only --experiment_name rnn_dorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs 1885 --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --experiment_name cnn_allorbitals_train_on_whole
python main.py $COMMON_CNN --max_epochs 1572 --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --keep_d_only --experiment_name cnn_dorbitals_train_on_whole
