# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import yaml
from modules.APViT import APViT
from timm.data import Mixup, create_transform

class APViTCifar10(nn.Module):
    def __init__(
        self,
        total_patches,
        hidden_channels,
        attn_embed_dim,
        num_transformer_layers,
        transformer_dropout
    ):
        super(APViTCifar10, self).__init__()
        self.vit = APViT(
            total_patches=total_patches,
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            transformer_dropout=transformer_dropout
        )

    def forward(self, x):
        return self.vit(x)

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dataloaders(batch_size, num_workers):
    train_transform = create_transform(
        input_size=32,
        is_training=True,
        auto_augment='rand-m5-mstd0.5',  # RandAugment with magnitude 9, matches DEiT setup
        re_prob=0.25,  # Random Erasing with probability 0.25 (DEiT setup)
        re_mode='pixel',  # Erase pixels
        re_count=1,  # Number of erasing operations
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 normalization
        std=[0.2470, 0.2435, 0.2616]
    )

    test_transform = create_transform(
        input_size=32,
        is_training=False,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True, download=True, transform=train_transform
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
        device,
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
    mixup_fn = Mixup(
        mixup_alpha=0.8,  # Mixup alpha for DeiT setup
        cutmix_alpha=1.0,  # CutMix alpha for DeiT setup
        prob=1.0,  # Probability of applying either
        switch_prob=0.5,  # Probability to switch between Mixup and CutMix
        mode='batch',  # Apply augmentations at the batch level
        label_smoothing=0.1,  # Label smoothing for DeiT
        num_classes=10  # Assuming CIFAR-10 dataset
    )

    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            #images, labels = mixup_fn(images, labels)
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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("hparams_config.yaml")

    batch_size = config.get("batch_size", 256)
    num_epochs = config.get("epochs", 200)
    lr = 0.0005 * batch_size / 512
    warmup_epochs = config.get("warmup_epochs", 5)
    weight_decay = config.get("weight_decay", 0.05)
    eta_min = config.get("eta_min", 1e-6)
    hidden_channels = config.get("hidden_channels", 32)
    attn_embed_dim = config.get("attn_embed_dim", 192)
    num_transformer_layers = config.get("num_transformer_layers", 6)
    transformer_dropout = config.get("transformer_dropout", 0.1)

    trainloader, testloader = get_dataloaders(batch_size, num_workers=4)

    patches_tests = [16, 14, 12, 10, 8]
    for total_patches in patches_tests:

        model = APViTCifar10(
            total_patches,
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            transformer_dropout=transformer_dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs-warmup_epochs,
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
                best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}

            with open(f"experiments/training_data/apvit_cifar10_{total_patches}.txt", "a") as file:
                file.write(f"{train_loss},{test_loss},{accuracy}\n")

        print(f"Best Accuracy: {best_accuracy:.4f}")
        torch.save(best_weights, f"models/apvit_cifar10_{total_patches}.pth")

if __name__ == "__main__": main()
