import json
import numpy as np
import torch.optim as optim
from src.evaluation import evaluate_model
from torch.optim.lr_scheduler import StepLR
import copy
import torch


def l2_regularization(model, lambda_l2):
    l2_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return lambda_l2 * l2_loss


def train_model(model, train_loader, val_loader, lr, max_epochs, device, l2_lambda, criterion, writer=None, train_on_whole_data=False, history_path=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model = copy.deepcopy(model)
    best_val_loss = np.inf
    best_epoch = 0
    early_stopping_mercy = max_epochs if train_on_whole_data else 500
    history = {"train_loss": [], "val_loss": []}
    if train_on_whole_data:
        print(f"III Training on whole data. Early stopping disabled. Running all {max_epochs} epochs.")
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    model.to(device)

    for epoch in range(max_epochs):
        total_loss = 0
        total_count = 0

        for X_batch, y_batch, identifiers_batch, features_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            features_batch = features_batch.to(device)
            output, attn_outputs = model(X_batch, features_batch, return_attn=True)
            if "channel_matrix" in attn_outputs:
                loss = criterion(output, y_batch, A=attn_outputs['channel_matrix'])
            else:
                loss = criterion(output, y_batch)

            if l2_lambda > 0:
                l2_loss = l2_regularization(model, lambda_l2=l2_lambda)
                loss += l2_loss
            if "channel" in attn_outputs:
                l1_penalty = attn_outputs.get("channel_l1", 0.0)
                loss += l1_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            total_count += len(X_batch)
        total_loss /= total_count
        val_loss, val_r2, _, _, _ = evaluate_model(model=model, dataloader=val_loader, device=device, criterion=criterion)
        history["train_loss"].append(float(total_loss))
        history["val_loss"].append(float(val_loss))
        scheduler.step()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}/{max_epochs}, LR: {scheduler.get_last_lr()[0]:.6f} train_loss: {total_loss:.4f}, val_loss: {val_loss:.4f} ({best_val_loss:.4f}) val_r2: {val_r2}"
            )

        if writer and len(val_loader) >1:
            writer.add_scalar("Loss/train", total_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for i, value in enumerate(val_r2):
                writer.add_scalar(f"R2/val_axis_{i}", value, epoch)

        # early stopping
        if train_on_whole_data:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > early_stopping_mercy:
            print("Early Stopping!!")
            if history_path:
                with open(history_path, "w") as f:
                    json.dump(history, f)
            return best_model, epoch

    if history_path:
        with open(history_path, "w") as f:
            json.dump(history, f)
    return best_model, epoch
