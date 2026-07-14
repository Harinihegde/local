import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PATIENCE, DEVICE, DATA_ROOT
from dataset import build_dataframe, DensityDataset, get_transforms
from model import DensityClassifier


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    return running_loss / len(loader.dataset), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    return running_loss / len(loader.dataset), 100. * correct / total, all_preds, all_labels


def main():
    print(f"Using device: {DEVICE}")
    df = build_dataframe(DATA_ROOT)
    print(f"Total images: {len(df)}")

    train_transform, val_transform = get_transforms()
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'test']

    train_loader = DataLoader(DensityDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(DensityDataset(val_df,   val_transform),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DensityClassifier(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{NUM_EPOCHS}\n{'='*60}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc}, 'best_density_model.pth')
            print("✓ Best model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete. History saved to training_history.json")


if __name__ == '__main__':
    main()