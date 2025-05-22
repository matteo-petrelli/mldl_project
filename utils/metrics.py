import torch
import matplotlib.pyplot as plt
import os
import csv

def evaluate_model(model, dataloader, device):
    """Evaluate accuracy on the given dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def plot_and_save_metrics(history, save_dir):
    """Plot training history and save it to file."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
    plt.close()

    # Save raw data as CSV
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_acc", "test_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_acc"][i],
                history["test_acc"][i]
            ])
