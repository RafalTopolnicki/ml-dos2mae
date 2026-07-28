import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
from src.io import write_dict_to_json


def evaluate_model(model, dataloader, device, criterion):
    if len(dataloader.dataset) <= 1:
        # total_loss, score, total_preds, total_targets
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan]
    model.eval()
    model.to(device)
    with torch.no_grad():
        total_loss = 0
        total_count = 0
        total_preds = []
        total_targets = []
        total_indentifier = []
        for X_batch, y_batch, indentifier_batch, features_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            features_batch = features_batch.to(device)
            output, attn_outputs = model(X_batch, features_batch, return_attn=True)
            if "channel_matrix" in attn_outputs:
                loss = criterion(output, y_batch, A=attn_outputs['channel_matrix'])
            else:
                loss = criterion(output, y_batch)
            total_loss += loss.item() * len(X_batch)
            total_count += len(X_batch)
            total_preds += output.detach().cpu().tolist()
            total_indentifier += list(indentifier_batch)
            total_targets += y_batch.detach().cpu().tolist()
    total_loss /= total_count
    score = r2_score(total_targets, total_preds, multioutput="raw_values")
    model.train()
    return total_loss, score, total_preds, total_targets, total_indentifier


def write_evaluation(model, dataloader, device, filepath, targets_names, criterion, scaler=None):
    total_loss, scores, total_preds, total_targets, total_identifiers = evaluate_model(
        model=model, dataloader=dataloader, device=device, criterion=criterion,
    )
    if np.isnan(total_preds).any():
        output = {
            "total_loss": np.nan,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "preds": np.nan,
            "targets": np.nan,
            "identifiers": np.nan,
        }
    else:
        # scale back the predictions
        if scaler:
            total_preds = scaler.inverse_transform(total_preds).tolist()
            total_targets = scaler.inverse_transform(total_targets).tolist()
        mae = mean_absolute_error(total_targets, total_preds)
        mape = mean_absolute_percentage_error(total_targets, total_preds)
        mse = mean_squared_error(total_targets, total_preds)
        rmse = np.sqrt(mse)
        scores = r2_score(total_targets, total_preds, multioutput="raw_values").tolist()
        scores_dict = dict(zip(targets_names, scores))
        output = {
            "total_loss": total_loss,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "r2": scores_dict,
            "preds": total_preds,
            "targets": total_targets,
            "identifiers": total_identifiers,
        }
    write_dict_to_json(output, filepath)
    return scores
