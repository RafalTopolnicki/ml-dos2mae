import argparse
import torch

from src.data import process_database, MAEDOSDataset
from src.evaluation import write_evaluation
from src.model import BiGRUModel
from src.cnnmodel import SimpleCNNModel, ResNet1dModel, DenseNet1dModel, VGG1dModel
from torch.utils.data import DataLoader
from src.loss import MSEWithPairAttentionLoss
import json
import os
import pickle


def predict(args):
    database_path = args["database"]
    dataset_path = args["datasetpath"]
    if dataset_path == "":
        dataset_path = os.path.dirname(database_path)
    batch_size = args["batch_size"]
    output_path = os.path.join(args["model_path"], args["output"])

    assert output_path.endswith(".json")

    experiment_args = json.load(open(os.path.join(args["model_path"], "arguments.json")))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>>> Will run on device {device}")
    print(f">>>> Outputs will be written to {output_path}")

    # read scaler
    scaler_path = os.path.join(args["model_path"], "scaler.pickle")
    print(scaler_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    doss, n_features, targets_dim, targets_names = process_database(
        path=args["database"],
        dataset_path=dataset_path,
        shuffle=False,
        keep_d_only=experiment_args["keep_d_only"],
        split_at_fermi=experiment_args["split_at_fermi"],
        keepE=experiment_args["keepE"],
        input_occupancy=experiment_args["input_occupancy"],
        energy_window_low=experiment_args["energy_window_low"],
        energy_window_high=experiment_args["energy_window_high"],
        energy_window_interpolation_pts=experiment_args["energy_window_interpolation_pts"],
        add_nitrogen=experiment_args["add_nitrogen"],
        use_grad=experiment_args["use_grad"],
        smooth_sigma=experiment_args["smooth_sigma"],
        target_label=experiment_args["target"],
    )
    dataset = MAEDOSDataset(list_data_dict=doss, scaler=scaler)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    assert experiment_args['loss_pair_weight'] == 0
    CRITERION = MSEWithPairAttentionLoss(pair_weight=0,
                                         preferred_matrix=None)

    input_dim = doss[0]["dos"].shape[-1]
    rnn_type = experiment_args["rnn_type"]
    assert rnn_type in ["bigru", "cnn", "resnet", "densenet", "vgg"]

    hidden_size   = experiment_args["hidden_size"]
    dropout_prob  = experiment_args["dropout_prob"]
    use_layernorm = experiment_args["use_layernorm"]
    cnn_depth     = experiment_args.get("cnn_depth")
    if cnn_depth is None:
        cnn_depth = {"cnn": 2, "resnet": 4, "densenet": 3, "vgg": 3}.get(rnn_type, 2)

    if rnn_type == "bigru":
        model = BiGRUModel(
            hidden_size=hidden_size,
            input_size=input_dim,
            num_layers=experiment_args["lstm_layers"],
            dropout_prob=dropout_prob,
            use_channel_attention=experiment_args['use_channel_attention'],
            use_temporal_attention=experiment_args['use_temporal_attention'],
            target_dim=targets_dim,
            use_channel_self_attention=experiment_args['use_channel_self_attention'],
            channel_self_attn_dim=experiment_args['channel_self_attn_dim'],
            attention_bias_matrix=experiment_args['attention_bias_matrix_path'],
            bias_strength=1.0,
            bidirectional=not experiment_args.get('one_direction', False),
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
    model.load_state_dict(torch.load(os.path.join(args["model_path"], "best_model.pth"), weights_only=True))
    # model, dataloader, criterion, device, filepath
    write_evaluation(
        model=model, dataloader=loader, device=device, filepath=output_path, targets_names=targets_names, scaler=scaler, criterion=CRITERION
    )


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
    parser.add_argument("--model_path", type=str, default="", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    predict(args)


if __name__ == "__main__":
    main()
