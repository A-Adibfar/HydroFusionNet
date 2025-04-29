import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(model, val_loader, device="cuda"):
    model.load_state_dict(torch.load("best_classifier_model.pth"))
    model = model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            feature1, feature2, feature3, feature4, diff_features, month, label = [
                b.to(device) for b in batch
            ]
            outputs = model(
                feature1, feature2, feature3, feature4, diff_features, month
            )
            preds.append(outputs.cpu())
            targets.append(label.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    pred_labels = (preds > 0.5).int()

    print("\nClassification Report:\n")
    print(classification_report(targets, pred_labels, digits=4))


def save_classification_results(
    model, val_loader, results_dir="classifierResults/", device="cuda"
):
    os.makedirs(results_dir, exist_ok=True)
    model.load_state_dict(torch.load("best_classifier_model.pth"))
    model = model.to(device)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            feature1, feature2, feature3, feature4, diff_features, month, label = [
                b.to(device) for b in batch
            ]
            outputs = model(
                feature1, feature2, feature3, feature4, diff_features, month
            )
            preds.append(outputs.cpu())
            targets.append(label.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    pred_labels = (preds > 0.5).int()

    cls_report = classification_report(targets, pred_labels, output_dict=True)
    pd.DataFrame(cls_report).transpose().to_csv(
        os.path.join(results_dir, "classification_report.csv")
    )

    cm = confusion_matrix(targets, pred_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(targets, preds)
    auc_score = roc_auc_score(targets, preds)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(targets, preds)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker=".")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "pr_curve.png"))
    plt.close()

    print(f"Results saved in {results_dir}/")
