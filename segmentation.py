import os
import gc
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm

from myDatasets import FeatureDataset, PatchedFeatureDataset
from decode_heads import SegmentationHead, PatchedSegmentationHead


def compute_iou(preds, labels, num_classes, device):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1).to(device)

    # Create mask for valid pixels (not ignore index -1)
    valid_mask = labels != -1
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    if len(labels) == 0:
        return [float('nan')] * num_classes

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious


def pixel_accuracy(preds, labels):
    # Create mask for valid pixels (not ignore index -1)
    valid_mask = labels != -1
    if valid_mask.sum() == 0:
        return 0.0

    labels = labels.to(preds.device)
    correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
    total = valid_mask.sum().item()

    return correct / total


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        # Ensure masks are long type for CrossEntropyLoss
        masks = masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, num_classes=19):
    model.eval()
    total_loss = 0.0
    total_iou = []
    total_acc = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            # Ensure masks are long type for CrossEntropyLoss
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            # Pass device to compute_iou and pixel_accuracy
            ious = compute_iou(preds, masks, num_classes, device)
            acc = pixel_accuracy(preds, masks)

            total_iou.append(ious)
            total_acc.append(acc)

    mean_iou = np.nanmean(np.array(total_iou), axis=0)
    overall_acc = np.mean(total_acc)
    return total_loss / len(dataloader), mean_iou, overall_acc


def run_training_loop(
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    epochs=5,
    lr=0.001,
    batch_size=128,
    device="cuda",
    num_workers=4
):

    # Clean cache
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize metric lists
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_mean_ious = []

    # Instantiate model, optimizer, loss
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, mean_iou, val_acc = evaluate(model, test_loader, loss_fn, device, 19)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_mean_ious.append(np.nanmean(mean_iou))

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Pixel Accuracy: {val_acc:.4f}")
        print(f"Val Mean IoU: {np.nanmean(mean_iou):.4f}")

    print("Done!")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_mean_ious': val_mean_ious
    }


if __name__ == "__main__":
    DATA_DIR = "cityscapes"
    FEAT_DIR = "cityscape_features_avg"
    P_FEAT_DIR = "cityscape_features_patched"
    METRIC_DIR = "train_metrics"
    CONF_PATH = "configs/repa_improved_ddt_xlen22de6_256.yaml"
    CKPT_PATH = "model.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TIME_STEPS = (0.95, 0.50)
    SPLITS = ("train", "test", "val")
    EPOCHS = 5
    L_RATE = 0.001
    B_SIZE = 128

    gc.collect()
    torch.cuda.empty_cache()


    for timestep in TIME_STEPS:
        ## Load data
        train_set = FeatureDataset(FEAT_DIR, DATA_DIR, split="train", timestep=timestep)
        test_set = FeatureDataset(FEAT_DIR, DATA_DIR, split="test", timestep=timestep)

        dataset = ConcatDataset([train_set, test_set])
        validation_set = FeatureDataset(FEAT_DIR, DATA_DIR, split="val", timestep=timestep)
        del train_set, test_set

        ## Get model
        model = SegmentationHead()

        metrics = run_training_loop(
            model=model,
            train_set=dataset,
            test_set=validation_set,
            epochs=EPOCHS,
            lr=L_RATE,
            batch_size=B_SIZE
        )

        # Save to JSON
        path = os.path.join(METRIC_DIR, f"metrics_{int(timestep*100)}_avg.json")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Access the metrics
        print("Training losses:", metrics['train_losses'])
        print("Validation losses:", metrics['val_losses'])
        print("Validation accuracies:", metrics['val_accuracies'])
        print("Validation mean IoUs:", metrics['val_mean_ious'])


    gc.collect()
    torch.cuda.empty_cache()


    for timestep in TIME_STEPS:
        ## Load data
        train_set = PatchedFeatureDataset(FEAT_DIR, DATA_DIR, split="train", timestep=timestep)
        test_set = PatchedFeatureDataset(FEAT_DIR, DATA_DIR, split="test", timestep=timestep)

        dataset = ConcatDataset([train_set, test_set])
        validation_set = PatchedFeatureDataset(FEAT_DIR, DATA_DIR, split="val", timestep=timestep)
        del train_set, test_set

        ## Get model
        model = PatchedSegmentationHead()

        metrics = run_training_loop(
            model=model,
            train_set=dataset,
            test_set=validation_set,
            epochs=EPOCHS,
            lr=L_RATE,
            batch_size=B_SIZE
        )

        # Save to JSON
        path = os.path.join(METRIC_DIR, f"metrics_{int(timestep*100)}_patched.json")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Access the metrics
        print("Training losses:", metrics['train_losses'])
        print("Validation losses:", metrics['val_losses'])
        print("Validation accuracies:", metrics['val_accuracies'])
        print("Validation mean IoUs:", metrics['val_mean_ious'])

    print("DONE")
