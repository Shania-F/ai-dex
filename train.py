from models import PokemonResNet
from data import get_pokemon_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def topk_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.sum().item() / target.size(0)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total = 0

    for images, labels in tqdm(dataloader, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_top1 += (outputs.argmax(1) == labels).sum().item()
        total_top5 += topk_accuracy(outputs, labels, 5) * images.size(0)
        total += images.size(0)

    return total_loss / total, total_top1 / total, total_top5 / total

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_top1 += (outputs.argmax(1) == labels).sum().item()
            total_top5 += topk_accuracy(outputs, labels, 5) * images.size(0)
            total += images.size(0)

    return total_loss / total, total_top1 / total, total_top5 / total

def main():
    data_dir = r"D:\PycharmProjects\Pokemon_Dataset_879"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader, dataset = get_pokemon_dataloaders(
        data_dir,
        batch_size=8,
        num_workers=2,
        val_split=0.1,
    )

    num_classes = len(dataset.classes)
    print(f"Classes: {num_classes}")

    model = PokemonResNet(num_classes=num_classes, pretrained=True, device=device)
    # model.summary()

    criterion = nn.CrossEntropyLoss()

    # ===== PHASE 1: FC only =====
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.model.fc.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.model.fc.parameters(), lr=1e-3)

    print("\n=== Phase 1: Train classifier only ===")
    for epoch in range(5):
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_top1, val_top5 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train: L={train_loss:.4f} T1={train_top1:.3f} T5={train_top5:.3f} | "
            f"Val: L={val_loss:.4f} T1={val_top1:.3f} T5={val_top5:.3f}"
        )

    # ===== PHASE 2: Fine-tune =====
    for p in model.model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.model.parameters(), lr=1e-4)

    print("\n=== Phase 2: Fine-tune full model ===")
    for epoch in range(9): # loss increases at 10
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_top1, val_top5 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train: L={train_loss:.4f} T1={train_top1:.3f} T5={train_top5:.3f} | "
            f"Val: L={val_loss:.4f} T1={val_top1:.3f} T5={val_top5:.3f}"
        )

    torch.save(model.state_dict(), "pokemon_resnet34.pth")
    print("Saved model to pokemon_resnet34.pth")

if __name__ == "__main__":
    main()
