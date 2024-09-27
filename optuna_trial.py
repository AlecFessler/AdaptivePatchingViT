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
import optuna
from optuna.exceptions import TrialPruned

class APViTCifar10(nn.Module):
    def __init__(
        self,
        num_patches,
        hidden_channels,
        attn_embed_dim,
        num_transformer_layers,
        stochastic_depth,
        scaling,
        max_scale,
        rotating
    ):
        super(APViTCifar10, self).__init__()
        self.vit = APViT(
            num_patches=num_patches,
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            stochastic_depth=stochastic_depth,
            scaling=scaling,
            max_scale=max_scale,
            rotating=rotating
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
        auto_augment=f'rand-m{augment_magnitude}-mstd0.5-inc1',  # RandAugment with magnitude 9, matches DEiT setup
        re_prob=re_prob,  # Random Erasing with probability 0.25 (DEiT setup)
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
        accumulation_steps,
        scaler,
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

    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    total_steps = len(train_loader)

    with tqdm(train_loader, unit="batch") as tepoch:
        for step, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            #images, labels = mixup_fn(images, labels)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            unscaled_loss = loss.item()

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tepoch.set_postfix(loss=unscaled_loss)
            running_loss += unscaled_loss

        if total_steps % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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

    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 256)
    warmup_epochs = config.get("warmup_epochs", 5)

    accumulation_steps = trial.suggest_int("accumulation_steps", 1, 4)
    lr_factor = trial.suggest_categorical("lr_factor", [256, 512])
    lr = 0.0005 * batch_size * accumulation_steps / lr_factor

    attn_embed_dim = trial.suggest_int("attn_embed_dim", 128, 512, step=64)
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 4, 8, step=2)
    hidden_channels = trial.suggest_int("hidden_channels", 16, 32, step=4)
    scaling = trial.suggest_categorical("scaling", ['isotropic', 'anisotropic', None])
    max_scale = trial.suggest_float("max_scale", 0.3, 0.4)
    rotating = trial.suggest_categorical("rotating", [True, False])

    stochastic_depth = trial.suggest_float("stochastic_depth", 0.0, 0.2)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    re_prob = trial.suggest_float("re_prob", 0.0, 0.5)
    augment_magnitude = trial.suggest_int("augment_magnitude", 1, 9)

    trainloader, testloader = get_dataloaders(batch_size, 4, augment_magnitude, re_prob)

    model = APViTCifar10(
        num_patches=16,
        hidden_channels=hidden_channels,
        attn_embed_dim=attn_embed_dim,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth,
        scaling=scaling,
        max_scale=max_scale,
        rotating=rotating
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': weight_decay}
    ], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)

    scaler = GradScaler()
    best_accuracy = 0.0

    for epoch in range(epochs):
            train_loss = train(model, trainloader, criterion, optimizer, scheduler, warmup_scheduler, warmup_epochs, epoch, accumulation_steps, scaler, device)
            test_loss, accuracy = evaluate(model, testloader, criterion, device)

            trial.report(accuracy, epoch)

            print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

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

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="APViT_Cifar10", storage="sqlite:///APViT_Cifar10.db", load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print("Best hyperparameters: ", trial.params)

    with open("best_trial.yaml", "w") as file:
        yaml.dump(trial.params, file)

if __name__ == "__main__": main()
