# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm
from modules.PatchEmbed import PatchEmbed
from modules.SelfAttn import SelfAttn

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=32,
            patch_size=8,
            in_chans=3,
            embed_dim=256 + 32
        )
        self.pos_embeds = nn.Parameter(torch.randn(1, 32*32 // 64, 256 + 32))

        self.cls_token = nn.Parameter(torch.randn(1, 1, 256 + 32))

        self.transformer1 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )
        self.transformer2 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )
        self.transformer3 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )
        self.transformer4 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )
        self.transformer5 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )
        self.transformer6 = SelfAttn(
            embed_dim=256 + 32,
            num_heads=4,
            dropout=0.1
        )

        self.norm = nn.LayerNorm(256 + 32)
        self.fc = nn.Linear(256 + 32, 10)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embeds
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2).contiguous()
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        x = self.transformer4(x)
        x = self.transformer5(x)
        x = self.transformer6(x)
        x = x[0]
        x = self.norm(x)
        x = self.fc(x)
        return x

def evaluate(
        model,
        test_loader,
        criterion,
        device
    ):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    saved_image = False

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if not saved_image:
                model.dynamic_patch.save_data = True
                saved_image = True

    accuracy = correct / total
    return running_loss / len(test_loader), accuracy

def train(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        warmup_scheduler,
        epoch,
        device
    ):
    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tepoch.set_postfix(loss=loss.item())
            running_loss += loss.item()

    if epoch < 5:
        warmup_scheduler.step()
    else:
        scheduler.step()

    return running_loss / len(train_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )

    num_epochs = 300
    model = CIFAR10Model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=40,
        T_mult=2,
        eta_min=0
    )
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: epoch / 5
    )

    best_weights = None
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        train_loss = train(
            model,
            trainloader,
            criterion,
            optimizer,
            scheduler,
            warmup_scheduler,
            epoch,
            device
        )
        test_loss, accuracy = evaluate(
            model,
            testloader,
            criterion,
            device
        )
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_weights, "std_vit_cifar10.pth")

if __name__ == "__main__":
    main()
