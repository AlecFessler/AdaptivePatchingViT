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
from modules.ViT import ViT
from timm.data import Mixup, create_transform
import optuna
from optuna.exceptions import TrialPruned

class STD_ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        num_patches=16,
        patch_size=8,
        in_channels=3,
        embed_dim=256,
        attn_heads=4,
        num_transformer_layers=6,
        stochastic_depth=0.1
    ):
        super(STD_ViT, self).__init__()
        self.vit = ViT(
            img_size=img_size,
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            num_transformer_layers=num_transformer_layers,
            stochastic_depth=stochastic_depth
        )

    def forward(self, x):
        return self.vit(x)

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
    running_vit_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            vit_loss = criterion(outputs, labels)
            running_vit_loss += vit_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    vit_test_loss = running_vit_loss / len(test_loader)

    return vit_test_loss, accuracy

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
        scaler,
        mixup_fn,
        device
    ):

    model.train()
    running_vit_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            images, labels = mixup_fn(images, labels)

            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(images)
                vit_loss = criterion(outputs, labels)

            if torch.isnan(vit_loss).any():
                raise TrialPruned()

            scaled_vit_loss = vit_loss / accumulation_steps
            scaler.scale(scaled_vit_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()

            running_vit_loss += vit_loss.item()
            tepoch.set_postfix(loss=running_vit_loss / (i + 1))

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    return running_vit_loss / len(train_loader)

def objective(trial):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("hparams_config.yaml")

    batch_size = config.get("batch_size", 256)
    accumulation_steps = config.get("accumulation_steps", 2)
    epochs = config.get("epochs", 100)
    warmup_epochs = config.get("warmup_epochs", 5)

    attn_embed_dim = config.get("attn_embed_dim", 256)
    num_transformer_layers = config.get("num_transformer_layers", 6)

    mixup_switch_prob = config.get("mixup_switch_prob", 0.5)
    label_smoothing = config.get("label_smoothing", 0.05)

    stochastic_depth = trial.suggest_float("stochastic_depth", 0.0, 0.2, step=0.05)
    re_prob = trial.suggest_float("re_prob", 0.0, 0.5, step=0.05)
    augment_magnitude = trial.suggest_int("augment_magnitude", 1, 9)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 1.0, step=0.1)
    cutmix_alpha = trial.suggest_float("cutmix_alpha", 0.1, 1.0, step=0.1)
    mixup_prob = trial.suggest_float("mixup_prob", 0.1, 1.0, step=0.1)

    lr = trial.suggest_float("lr", 0.0001, 0.001, log=True)
    lr_min = lr * 0.1
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.01, log=True)

    trainloader, testloader = get_dataloaders(
        batch_size,
        num_workers=4,
        augment_magnitude=augment_magnitude,
        re_prob=re_prob
    )

    model = STD_ViT(
        img_size=32,
        num_patches=16,
        patch_size=8,
        in_channels=3,
        embed_dim=attn_embed_dim,
        attn_heads=4,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
    ], lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs-warmup_epochs,
        eta_min=lr_min
    )

    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: epoch / warmup_epochs
    )

    scaler = GradScaler()

    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mixup_prob,
        switch_prob=mixup_switch_prob,
        mode='batch', label_smoothing=label_smoothing,
        num_classes=10
    )

    best_accuracy = 0.0

    for epoch in range(epochs):
        vit_train_loss = train(
            model,
            trainloader,
            criterion,
            optimizer,
            scheduler,
            warmup_scheduler,
            warmup_epochs,
            epoch,
            accumulation_steps,
            scaler,
            mixup_fn,
            device
        )
        vit_test_loss, accuracy = evaluate(
            model,
            testloader,
            criterion,
            device
        )

        print(f"Epoch: {epoch + 1}/{epochs} | ViT Train Loss: {vit_train_loss:.4f} | ViT Test Loss: {vit_test_loss:.4f} | Accuracy: {accuracy*100:.2f}%")

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise TrialPruned()

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return best_accuracy


def main():
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=10,
        interval_steps=1
    )

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="APViT_Cifar10_AP", storage="sqlite:///APViT_Cifar10_AP.db", load_if_exists=True)
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print("Best hyperparameters: ", trial.params)

    with open("best_trial.yaml", "w") as file:
        yaml.dump(trial.params, file)

if __name__ == "__main__": main()
