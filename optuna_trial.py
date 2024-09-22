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
import optuna
from optuna import TrialPruned

class DpsViTCifar10(nn.Module):
    def __init__(
        self,
        hidden_channels,
        attn_embed_dim,
        num_transformer_layers,
        stn_dropout,
        patch_dropout,
        transformer_dropout
    ):
        super(DpsViTCifar10, self).__init__()
        self.vit = DpsViT(
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            stn_dropout=stn_dropout,
            patch_dropout=patch_dropout,
            transformer_dropout=transformer_dropout,
        )

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
    accumulation_steps,
    device
):
    scaler = GradScaler()
    model.train()
    running_loss = 0.0

    with tqdm(enumerate(train_loader), total=len(train_loader), unit="batch") as tepoch:
        for i, (images, labels) in tepoch:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tepoch.set_postfix(loss=loss.item() * accumulation_steps)
            running_loss += loss.item() * accumulation_steps

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    return running_loss / len(train_loader)

def objective(trial):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("hparams_config.yaml")

    # Fixed parameters
    batch_size = config.get("batch_size", 256)
    accumulation_steps = 1
    attn_embed_dim = config.get("attn_embed_dim", 256)
    num_transformer_layers = config.get("num_transformer_layers", 4)
    stn_dropout = config.get("stn_dropout", 0.0)
    patch_dropout = config.get("patch_dropout", 0.0)
    transformer_dropout = config.get("transformer_dropout", 0.4)
    label_smoothing = config.get("label_smoothing", 0.05)

    # Trial parameters
    dynamic_patch_lr = trial.suggest_float("dynamic_patch_lr", 1e-5, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    t_0 = trial.suggest_int("T_0", 10, 75)
    t_mult = trial.suggest_int("T_mult", 1, 3)
    eta_min = trial.suggest_float("eta_min", 0, 1e-3, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 5, 20)

    hidden_channels = trial.suggest_int("hidden_channels", 32, 128, step=8)

    trainloader, testloader = get_dataloaders(batch_size, 4)

    model = DpsViTCifar10(
        hidden_channels=hidden_channels,
        attn_embed_dim=attn_embed_dim,
        num_transformer_layers=num_transformer_layers,
        stn_dropout=stn_dropout,
        patch_dropout=patch_dropout,
        transformer_dropout=transformer_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW([
        {"params": model.vit.dynamic_patch.parameters(), "lr": dynamic_patch_lr},
        {'params': [p for n, p in model.named_parameters() if 'dynamic_patch' not in n]},
    ], lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)
    best_accuracy = 0.0

    for epoch in range(100):
            train_loss = train(model, trainloader, criterion, optimizer, scheduler, warmup_scheduler, warmup_epochs, epoch, accumulation_steps, device)
            test_loss, accuracy = evaluate(model, testloader, criterion, device)

            trial.report(accuracy, epoch)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss} | Test Loss: {test_loss} | Accuracy: {accuracy}")

            if trial.should_prune():
                raise TrialPruned()

            if accuracy > best_accuracy:
                best_accuracy = accuracy


    return best_accuracy

def main():
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=20,
        interval_steps=1
    )

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="DpsViT_Cifar10_2", storage="experiments/sqlite:///DpsViT_Cifar10_2.db", load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print("Best hyperparameters: ", trial.params)

    with open("best_trial.yaml", "w") as file:
        yaml.dump(trial.params, file)

if __name__ == "__main__": main()
