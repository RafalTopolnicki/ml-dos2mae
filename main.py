import argparse

import torch

from src.data import process_database, train_val_test_split, MAEDOSDataset
from src.training import train_model
from src.evaluation import write_evaluation
from src.utils import seed_everything, create_experiment_dir, experiment_name_with_timestamp, average_dict_values
from src.model import BiGRUModel
from src.cnnmodel import SimpleCNNModel, ResNet1dModel, DenseNet1dModel, VGG1dModel
from torch.utils.tensorboard import SummaryWriter
from src.io import write_dict_to_json, write_model
from torch.utils.data import DataLoader
from src.consts import DOS_INTERPOLATION_GRID_PTS, POSSIBLE_TARGETS
from src.loss import MSEWithPairAttentionLoss
import pandas as pd
import os
import pickle
import numpy as np
from copy import deepcopy


def run_experiment(args):
    database_path = args["database"]
    dataset_path = args["datasetpath"]
    if dataset_path == "":
        dataset_path = os.path.dirname(database_path)
    batch_size = args["batch_size"]
    lstm_layers = args["lstm_layers"]
    hidden_size = args["hidden_size"]
    max_epochs = args["max_epochs"]
    n_limit = args["n_limit"]
    dropout_prob = args["dropout_prob"]
    rnn_type = args["rnn_type"]
    keep_d_only = args["keep_d_only"]
    split_at_fermi = args["split_at_fermi"]
    keepE = args["keepE"]
    use_temporal_attention = args["use_temporal_attention"]
    use_channel_attention = args["use_channel_attention"]
    use_channel_self_attention = args["use_channel_self_attention"]
    channel_self_attn_dim = args["channel_self_attn_dim"]
    one_direction = args["one_direction"]
    use_layernorm = args["use_layernorm"]
    lr = args["lr"]
    l2_lambda = args["l2_lambda"]
    seed = args["seed"]
    cv_folds = args["cv_folds"]
    folds = args["folds"]
    save_model = args["save_model"]
    train_on_whole_data = args["train_on_whole_data"]
    input_occupancy = args["input_occupancy"]
    energy_window_low = args["energy_window_low"]
    energy_window_high = args["energy_window_high"]
    energy_window_interpolation_pts = args["energy_window_interpolation_pts"]
    smooth_sigma = args["smooth_sigma"]
    add_nitrogen = args["add_nitrogen"]
    use_grad = args["use_grad"]
    use_pt = args["use_pt"]
    target_label = args["target"]
    attention_bias_matrix_path = args["attention_bias_matrix_path"]
    attention_bias_strength = args["attention_bias_strength"]
    loss_pair_weight = args["loss_pair_weight"]
    noise_height_fraction = args["noise_height_fraction"]
    cnn_depth = args["cnn_depth"]
    if cnn_depth is None:
        cnn_depth = {"cnn": 2, "resnet": 4, "densenet": 3, "vgg": 3}.get(rnn_type, 2)

    seed_everything(seed)
    experiment_name = experiment_name_with_timestamp(args["experiment_name"])

    attention_bias_matrix = None
    if attention_bias_matrix_path is not None:
        attention_bias_matrix = pd.read_csv(attention_bias_matrix_path, sep=',', header=None).to_numpy()
        attention_bias_matrix = np.abs(attention_bias_matrix)

    loss_pair_weight = np.abs(loss_pair_weight)
    if loss_pair_weight > 0 and attention_bias_matrix is None:
        raise ValueError('attention_bias_matrix required when loss_pair_weight >0')

    if loss_pair_weight > 0:
        CRITERION = MSEWithPairAttentionLoss(pair_weight=loss_pair_weight,
                                         preferred_matrix=attention_bias_matrix*loss_pair_weight)
    else:
        CRITERION = MSEWithPairAttentionLoss(pair_weight=0,
                                             preferred_matrix=None)

    if attention_bias_matrix is not None:
        attention_bias_matrix = attention_bias_matrix * attention_bias_strength

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    doss, n_features, targets_dim, targets_names = process_database(
        path=database_path,
        dataset_path=dataset_path,
        shuffle=True,
        keep_d_only=keep_d_only,
        keepE=keepE,
        split_at_fermi=split_at_fermi,
        input_occupancy=input_occupancy,
        energy_window_low=energy_window_low,
        energy_window_high=energy_window_high,
        energy_window_interpolation_pts=energy_window_interpolation_pts,
        smooth_sigma=smooth_sigma,
        add_nitrogen=add_nitrogen,
        use_grad=use_grad,
        use_pt=use_pt,
        n_limit=n_limit,
        target_label=target_label,
        noise_height_fraction=noise_height_fraction,
    )
    assert n_features >= 1

    input_dim = doss[0]["dos"].shape[-1]
    features_dim = doss[0]["features"].shape[-1]
    fold_test_scores = {}
    experiment_path_parent = os.path.join(args["output"], experiment_name)

    for fold_id, X_train, X_val, X_test in train_val_test_split(
        doss, cv_folds=cv_folds, which_folds=folds, train_on_whole_data=train_on_whole_data
    ):
        experiment_path = create_experiment_dir(args["output"], experiment_name, fold_id)
        write_dict_to_json(args, os.path.join(experiment_path, "arguments.json"))
        print(f">>>> Fold {fold_id}")
        print(f">>>> Will run on device {device}")
        print(f">>>> Outputs will be written to {experiment_path}")

        train_dataset = MAEDOSDataset(list_data_dict=X_train, scaler=None)
        scaler = deepcopy(train_dataset.scaler)
        if train_on_whole_data:
            val_dataset  = train_dataset
            test_dataset = train_dataset
        else:
            val_dataset  = MAEDOSDataset(list_data_dict=X_val,  scaler=scaler)
            test_dataset = MAEDOSDataset(list_data_dict=X_test, scaler=scaler)

        # save scaler to the file
        if save_model:
            with open(os.path.join(experiment_path, "scaler.pickle"), "wb") as f:
                pickle.dump(scaler, f)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"III Input dim is: DOS {input_dim} Features: {features_dim}")
        print(f"III Output dim is: {targets_dim}")
        print(f"III Datasetsizes. train: {len(X_train)} val: {len(X_test)} test: {len(X_test)}")
        if rnn_type == "lstm":
            raise ValueError('rnn_type=lstm not supported')
        elif rnn_type == "bilstm":
            raise ValueError('rnn_type=bilstm not supported')
        elif rnn_type == "bigru":
            model = BiGRUModel(
                hidden_size=hidden_size,
                input_size=input_dim,
                num_layers=lstm_layers,
                dropout_prob=dropout_prob,
                use_layernorm=use_layernorm,
                use_temporal_attention=use_temporal_attention,
                use_channel_attention=use_channel_attention,
                n_features=n_features,
                target_dim=targets_dim,
                use_channel_self_attention=use_channel_self_attention,
                channel_self_attn_dim=channel_self_attn_dim,
                attention_bias_matrix=attention_bias_matrix,
                bias_strength=1.0, # attention_bias_matrix is multipied earlier
                bidirectional=not one_direction,
            )
        elif rnn_type == "cnn":
            model = SimpleCNNModel(
                input_size=input_dim,
                n_features=n_features,
                target_dim=targets_dim,
                hidden_channels=hidden_size,
                dropout_prob=dropout_prob,
                use_layernorm=use_layernorm,
                n_layers=cnn_depth,
            )
        elif rnn_type == "resnet":
            model = ResNet1dModel(
                input_size=input_dim,
                n_features=n_features,
                target_dim=targets_dim,
                hidden_channels=hidden_size,
                dropout_prob=dropout_prob,
                use_layernorm=use_layernorm,
                n_blocks=cnn_depth,
            )
        elif rnn_type == "densenet":
            model = DenseNet1dModel(
                input_size=input_dim,
                n_features=n_features,
                target_dim=targets_dim,
                hidden_channels=hidden_size,
                dropout_prob=dropout_prob,
                use_layernorm=use_layernorm,
                n_blocks=cnn_depth,
            )
        elif rnn_type == "vgg":
            model = VGG1dModel(
                input_size=input_dim,
                n_features=n_features,
                target_dim=targets_dim,
                hidden_channels=hidden_size,
                dropout_prob=dropout_prob,
                use_layernorm=use_layernorm,
                n_blocks=cnn_depth,
            )
        else:
            raise ValueError(f"Unknown RNN: {rnn_type}")

        writer = SummaryWriter(os.path.join(experiment_path, "tensorboard"))
        model, total_epochs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            max_epochs=max_epochs,
            device=device,
            writer=writer,
            l2_lambda=l2_lambda,
            criterion=CRITERION,
            train_on_whole_data=train_on_whole_data,
        )

        # model, dataloader, criterion, device, filepath
        _ = write_evaluation(
            model=model,
            dataloader=train_loader,
            device=device,
            filepath=os.path.join(experiment_path, "eval_train.json"),
            targets_names=targets_names,
            scaler=scaler,
            criterion=CRITERION,
        )
        if not train_on_whole_data:
            _ = write_evaluation(
                model=model,
                dataloader=val_loader,
                device=device,
                filepath=os.path.join(experiment_path, "eval_val.json"),
                targets_names=targets_names,
                scaler=scaler,
                criterion=CRITERION,
            )
            test_score = write_evaluation(
                model=model,
                dataloader=test_loader,
                device=device,
                filepath=os.path.join(experiment_path, "eval_test.json"),
                targets_names=targets_names,
                scaler=scaler,
                criterion=CRITERION,
            )
            fold_test_scores[fold_id] = test_score
        write_dict_to_json(
            {"total_epochs": total_epochs} | {"target_cols": targets_names},
            os.path.join(experiment_path, "summary.json"),
        )
        if save_model:
            write_model(model, os.path.join(experiment_path, "best_model.pth"))
    average_score = average_dict_values(fold_test_scores)
    if train_on_whole_data == False:
        fold_test_scores["average_score"] = dict(zip(targets_names, average_score))
        write_dict_to_json(fold_test_scores, os.path.join(experiment_path_parent, "folds_summary.json"))


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True, help="Path to the database file")
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="",
        help="Path to directory where data is strored. If not given, the ath is infered from the --database path",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--experiment_name", type=str, default="", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--target", choices=POSSIBLE_TARGETS, type=str, default="MAE")
    parser.add_argument("--cv_folds", type=int, default=8, help="Number of CV folds")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7], help="Which folds to use")
    parser.add_argument("--n_limit", type=int, default=0, help="Limit number of structures to train on. 0 for no limit")
    parser.add_argument("--max_epochs", type=int, default=5000, help="Maximal number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--l2_lambda", type=float, default=0.0, help="L2 penalty coef")
    parser.add_argument("--rnn_type", choices=["lstm", "bilstm", "bigru", "cnn", "resnet", "densenet", "vgg"], default="bigru")
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--hidden_size", type=int, default=50, help="Hidden layers in LSTM")
    parser.add_argument("--dropout_prob", type=float, default=0.3, help="Dropout prob")
    parser.add_argument("--energy_window_low", type=float, default=-4.0)
    parser.add_argument("--energy_window_high", type=float, default=4.0)
    parser.add_argument("--energy_window_interpolation_pts", type=int, default=DOS_INTERPOLATION_GRID_PTS)
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--keepE", action="store_true")
    parser.add_argument("--split_at_fermi", action="store_true")
    parser.add_argument("--keep_d_only", action="store_true")
    parser.add_argument("--one_direction", action="store_true")
    parser.add_argument("--use_channel_attention", action="store_true")
    parser.add_argument("--use_channel_self_attention", action="store_true")
    parser.add_argument("--use_temporal_attention", action="store_true")
    parser.add_argument("--channel_self_attn_dim", type=int, default=32, help="Self attention channel dim")
    parser.add_argument("--attention_bias_matrix_path", type=str, default=None, help="Path to file containing attention bias matrix")
    parser.add_argument("--attention_bias_strength", type=float, default=1.0, help="Added to self attention mechanism")
    parser.add_argument("--loss_pair_weight", type=float, default=0.00, help="Added to loss function")
    parser.add_argument("--noise_height_fraction", type=float, default=0.0, help="Noise added to DOS")
    parser.add_argument("--cnn_depth", type=int, default=None, help="Depth of CNN models: n_layers (cnn) or n_blocks (resnet/densenet/vgg). Default: 2/4/3/3 per arch")
    parser.add_argument("--use_layernorm", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--train_on_whole_data", action="store_true")
    parser.add_argument("--input_occupancy", action="store_true")
    parser.add_argument("--use_grad", action="store_true")
    parser.add_argument("--use_pt", action="store_true")
    parser.add_argument("--add_nitrogen", action="store_true")

    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
