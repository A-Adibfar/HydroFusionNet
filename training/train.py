import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from models.early_stopping import EarlyStopping


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device="cuda"):
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    early_stopper = EarlyStopping(
        patience=20, delta=0.001, path="best_classifier_model.pth"
    )

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        preds, targets = [], []
        loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )

        for batch in loop:
            feature1, feature2, feature3, feature4, diff_features, month, label = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            outputs = model(
                feature1, feature2, feature3, feature4, diff_features, month
            )
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds.append(outputs.detach().cpu())
            targets.append(label.detach().cpu())
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation logic (optional to add here, or split into evaluate.py)
        early_stopper(sum(train_losses) / len(train_losses), model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
