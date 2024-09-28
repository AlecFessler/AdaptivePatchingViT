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
from modules.StandardViTPosAdd import StandardViTPosAdd
from timm.data import Mixup, create_transform

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dataloaders(
        batch_size,
        num_workers=2,
        augment_magnitude=9,
        re_prob=0.25
    ):
    train_transform = create_transform(
        input_size=32,
        is_training=True,
        auto_augment=f'rand-m{augment_magnitude}-mstd0.5-inc1',
        re_prob=re_prob,
        re_mode='pixel',
        re_count=1,
        mean=[0.4914, 0.4822, 0.4465],
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
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=10
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
    epochs = config.get("epochs", 200)
    warmup_epochs = config.get("warmup_epochs", 5)
    weight_decay = config.get("weight_decay", 0.000015)
    lr_factor = config.get("lr_factor", 512)
    lr = 0.0005 * batch_size / lr_factor
    eta_min = config.get("eta_min", 0.0001)
    attn_embed_dim = config.get("attn_embed_dim", 256)
    num_transformer_layers = config.get("num_transformer_layers", 8)
    stochastic_depth = config.get("stochastic_depth", 0.15)
    label_smoothing = config.get("label_smoothing", 0.05)
    re_prob = config.get("re_prob", 0.15)
    augment_magnitude = config.get("augment_magnitude", 5)

    trainloader, testloader = get_dataloaders(batch_size, num_workers=4, augment_magnitude=augment_magnitude, re_prob=re_prob)

    model = StandardViTPosAdd(
        img_size=32,
        patch_size=8,
        in_channels=3,
        attn_embed_dim=attn_embed_dim,
        attn_heads=4,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': weight_decay}
    ], lr=lr)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs-warmup_epochs,
        eta_min=eta_min
    )
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: epoch / warmup_epochs
    )

    best_weights = None
    best_accuracy = 0.0

    for epoch in range(epochs):
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

        print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}

        with open("experiments/training_data/standard_vit_cifar10.txt", "a") as file:
            file.write(f"{train_loss},{test_loss},{accuracy}\n")

    print(f"Best Accuracy: {best_accuracy:.4f}")
    torch.save(best_weights, "models/standard_vit_cifar10.pth")

if __name__ == "__main__":
    main()
