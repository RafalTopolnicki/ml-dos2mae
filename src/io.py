import json
import torch


def write_dict_to_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def write_model(model, filepath):
    torch.save(model.state_dict(), filepath)
