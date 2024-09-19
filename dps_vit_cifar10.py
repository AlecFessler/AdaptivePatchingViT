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
import yaml
from modules.DpsViT import DpsViT

class DpsViTCifar10(nn.Module):
    def __init__(self):
        super(DpsViTCifar10, self).__init__()
        self.vit = DpsViT()

    def forward(self, x):
        return self.vit(x)

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dataloaders(batch_size, num_workers):
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader

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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    test_loss = running_loss / len(test_loader)

    with open("dps_vit_test_loss.txt", "a") as file:
        file.write(f"{test_loss}\n")

    return test_loss, accuracy

def train(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        warmup_scheduler,
        warmup_epochs,
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

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    return running_loss / len(train_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("hparams_config.yaml")

    batch_size = config.get("batch_size", 256)
    num_workers = config.get("num_workers", 4)
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 1e-3)
    num_epochs = config.get("num_epochs", 300)
    t_0 = config.get("T_0", 40)
    t_mult = config.get("T_mult", 2)
    eta_min = config.get("eta_min", 0)
    warmup_epochs = config.get("warmup_epochs", 5)
    label_smoothing = config.get("label_smoothing", 0.1)

    trainloader, testloader = get_dataloaders(batch_size, num_workers)

    model = DpsViTCifar10().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t_0,
        T_mult=t_mult,
        eta_min=eta_min
    )
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: epoch / warmup_epochs
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
            warmup_epochs,
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

    torch.save(best_weights, "dps_vit_cifar10.pth")

if __name__ == "__main__": main()
