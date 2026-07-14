import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

from config import BATCH_SIZE, DEVICE, DATA_ROOT
from dataset import build_dataframe, DensityDataset, get_transforms
from model import DensityClassifier
from train import validate
import torch.nn as nn


def main():
    df = build_dataframe(DATA_ROOT)
    _, val_transform = get_transforms()
    val_df = df[df['split'] == 'test']
    val_loader = DataLoader(DensityDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DensityClassifier(num_classes=3).to(DEVICE)
    checkpoint = torch.load('best_density_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)

    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(val_labels, val_preds)

    print(f"\n{'='*60}\nFINAL EVALUATION METRICS\n{'='*60}")
    print(f"Accuracy:      {val_acc:.2f}%")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    metrics = {'accuracy': float(val_acc), 'precision': float(precision),
               'recall': float(recall), 'f1_score': float(f1), 'cohens_kappa': float(kappa)}
    with open('density_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    class_names = ['Sparse', 'Medium', 'Dense']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix - Crowd Density Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved confusion_matrix.png")


if __name__ == '__main__':
    main()