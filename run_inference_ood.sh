#!/bin/bash
cd "$(dirname "$0")"

OOD_DB="DATA/explicit_database.csv"

python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae/2026-07-28-030759_rnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae/2026-07-28-031155_rnn_dorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae/2026-07-28-031539_cnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae/2026-07-28-031721_cnn_dorbitals_train_on_whole/fold_0

python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc/2026-07-28-031834_rnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc/2026-07-28-032342_rnn_dorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc/2026-07-28-032801_cnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc/2026-07-28-033018_cnn_dorbitals_train_on_whole/fold_0

python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc+dorbitals/2026-07-28-033224_rnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc+dorbitals/2026-07-28-033711_rnn_dorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc+dorbitals/2026-07-28-034149_cnn_allorbitals_train_on_whole/fold_0
python evaluate.py --database ${OOD_DB} --output eval_explicit.json --model_path experiments/mae+esoc+dorbitals/2026-07-28-034449_cnn_dorbitals_train_on_whole/fold_0
