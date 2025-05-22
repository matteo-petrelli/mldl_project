from utils.metrics import evaluate_model, plot_and_save_metrics
import os
import torch

def train_model(train_data, val_data, test_data, lr=0.1, epochs=100, batch_size=128, output_dir="results"):
    """Train model and save plots/metrics to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    history = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader, device)
        test_acc = evaluate_model(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    plot_and_save_metrics(history, save_dir=output_dir)

    # Optionally save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    return model
