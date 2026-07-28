#!/bin/bash
cd "$(dirname "$0")"

COMMON_RNN="--database DATA/database.csv
  --batch_size 128 --seed 100
  --cv_folds 8 --folds 0 1 2 3 4 5 6 7
  --max_epochs 100000 --lr 0.001 --l2_lambda 0.0
  --hidden_size 100 --dropout_prob 0.3
  --energy_window_low -3.0 --energy_window_high 3.0
  --energy_window_interpolation_pts 200
  --keepE --split_at_fermi --use_layernorm --save_model
  --rnn_type bigru --lstm_layers 3
  --use_channel_self_attention --use_temporal_attention
  --channel_self_attn_dim 32"

COMMON_CNN="--database DATA/database.csv
  --batch_size 128 --seed 100
  --cv_folds 8 --folds 0 1 2 3 4 5 6 7
  --max_epochs 100000 --lr 0.001 --l2_lambda 0.0
  --hidden_size 100 --dropout_prob 0.3
  --energy_window_low -3.0 --energy_window_high 3.0
  --energy_window_interpolation_pts 200
  --keepE --split_at_fermi --use_layernorm --save_model
  --rnn_type cnn --cnn_depth 4 --noise_height_fraction 0.0"

python main.py $COMMON_RNN --output experiments/mae              --target MAE              --experiment_name rnn_allorbitals
python main.py $COMMON_RNN --output experiments/mae              --target MAE              --keep_d_only --experiment_name rnn_dorbitals
python main.py $COMMON_CNN --output experiments/mae              --target MAE              --experiment_name cnn_allorbitals
python main.py $COMMON_CNN --output experiments/mae              --target MAE              --keep_d_only --experiment_name cnn_dorbitals
#
python main.py $COMMON_RNN --output experiments/mae+esoc         --target MAE+ESOC         --experiment_name rnn_allorbitals
python main.py $COMMON_RNN --output experiments/mae+esoc         --target MAE+ESOC         --keep_d_only --experiment_name rnn_dorbitals
python main.py $COMMON_CNN --output experiments/mae+esoc         --target MAE+ESOC         --experiment_name cnn_allorbitals
python main.py $COMMON_CNN --output experiments/mae+esoc         --target MAE+ESOC         --keep_d_only --experiment_name cnn_dorbitals
#
python main.py $COMMON_RNN --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --experiment_name rnn_allorbitals
python main.py $COMMON_RNN --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --keep_d_only --experiment_name rnn_dorbitals
python main.py $COMMON_CNN --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --experiment_name cnn_allorbitals
python main.py $COMMON_CNN --output experiments/mae+esoc+dorbitals --target MAE+ESOC+DORBITALS --keep_d_only --experiment_name cnn_dorbitals
