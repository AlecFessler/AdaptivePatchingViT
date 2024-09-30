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
from modules.AdaptivePatchLoss import AdaptivePatchLoss
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
        pos_embed_size,
        scaling,
        max_scale,
        rotating
    ):
        super(APViTCifar10, self).__init__()
        self.vit = APViT(
            num_patches=num_patches,
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            pos_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            stochastic_depth=stochastic_depth,
            pos_embed_size=pos_embed_size,
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
    running_ap_loss = 0.0
    correct = 0
    total = 0

    ap_criterion, vit_criterion = criterion

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attn_weights, interpolated_pos_embeds = model(inputs)
            fixed_pos_embeds = model.vit.pos_embeds
            vit_loss = vit_criterion(outputs, labels)
            ap_loss = ap_criterion(attn_weights, fixed_pos_embeds, interpolated_pos_embeds)
            running_vit_loss += vit_loss.item()
            running_ap_loss += ap_loss
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    vit_test_loss = running_vit_loss / len(test_loader)
    ap_test_loss = running_ap_loss / len(test_loader)

    return vit_test_loss, ap_test_loss, accuracy

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
        ema_decay,
        mixup_fn,
        device
    ):

    ap_criterion, vit_criterion = criterion
    model.train()
    running_vit_loss = 0.0
    running_ap_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            images, labels = mixup_fn(images, labels)

            # zero optimizer to begin next set of mini-batch gradient accumulation
            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            # forward pass and compute loss for both ViT and Adaptive Patch modules
            with autocast(device_type=device.type):
                outputs, attn_weights, interpolated_pos_embeds = model(images)
                fixed_pos_embeds = model.vit.pos_embeds
                vit_loss = vit_criterion(outputs, labels)
                if torch.isnan(vit_loss).any():
                    raise ValueError("NaN loss")
                ap_loss = ap_criterion(attn_weights, fixed_pos_embeds, interpolated_pos_embeds)
                if torch.isnan(ap_loss).any():
                    raise ValueError("NaN loss")

            model.vit.requires_grad = False
            model.vit.adaptive_patches.requires_grad = True

            # Scale loss and backpropagate gradients for adaptive patches
            scaled_ap_loss = ap_loss / accumulation_steps
            scaler.scale(scaled_ap_loss).backward(retain_graph=True)

            model.vit.requires_grad = True
            model.vit.adaptive_patches.requires_grad = False

            # Scale loss and backpropagate gradients for ViT
            scaled_vit_loss = vit_loss / accumulation_steps
            scaler.scale(scaled_vit_loss).backward()

            # Step optimizer at the end of accumulation steps and update grad scaler
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Clone the current adaptive patching weights for EMA
                ap_weights = {k: v.clone().detach() for k, v in model.vit.adaptive_patches.state_dict().items()}

                scaler.step(optimizer)
                scaler.update()

                # Update EMA weights for adaptive patches
                for name, param in model.vit.adaptive_patches.named_parameters():
                    param.data = ema_decay * param.data + (1 - ema_decay) * ap_weights[name]

            # Update progress bar with loss
            tepoch.set_postfix(loss=running_vit_loss / (i + 1))
            running_vit_loss += vit_loss.item()
            running_ap_loss += ap_loss

    # Step appropriate scheduler based on epoch
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    return running_vit_loss / len(train_loader), running_ap_loss / len(train_loader)

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
    weight_decay = config.get("weight_decay", 0.00015)
    lr_factor = config.get("lr_factor", 512)
    lr = 0.0005 * accumulation_steps * batch_size / lr_factor
    eta_min = config.get("eta_min", 0.00015)
    hidden_channels = config.get("hidden_channels", 16)
    attn_embed_dim = config.get("attn_embed_dim", 256)
    num_transformer_layers = config.get("num_transformer_layers", 6)
    stochastic_depth = config.get("stochastic_depth", 0.15)
    re_prob = config.get("re_prob", 0.15)
    augment_magnitude = config.get("augment_magnitude", 5)
    mixup_alpha = config.get("mixup_alpha", 0.5)
    cutmix_alpha = config.get("cutmix_alpha", 0.2)
    mixup_prob = config.get("mixup_prob", 0.5)
    mixup_switch_prob = config.get("mixup_switch_prob", 0.5)
    label_smoothing = config.get("label_smoothing", 0.05)

    trainloader, testloader = get_dataloaders(
        batch_size,
        num_workers=4,
        augment_magnitude=augment_magnitude,
        re_prob=re_prob
    )

    model = APViTCifar10(
        num_patches=16,
        hidden_channels=hidden_channels,
        attn_embed_dim=attn_embed_dim,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth,
        pos_embed_size=4,
        scaling=None,
        max_scale=0.4,
        rotating=False
    ).to(device)

    attn_temperature = trial.suggest_float("attn_temperature", 0.1, 1.5)
    top_k_focus = trial.suggest_int("top_k_focus", 1, 6)
    attn_loss_weight = trial.suggest_float("attn_loss_weight", 0.5, 5.0, step=0.5)
    diversity_loss_weight = trial.suggest_float("diversity_loss_weight", 1.0, 5.0, step=0.5)

    ap_lr = trial.suggest_float("ap_lr", 0.0001, 0.001)
    ap_weight_decay = trial.suggest_float("ap_weight_decay", 1e-5, 1e-2)
    ema_decay = trial.suggest_float("ema_decay", 0.9, 0.999)

    print(f'Trial: {trial.number} started with\nattn_temperature: {attn_temperature}\ntop_k_focus: {top_k_focus}\nattn_loss_weight: {attn_loss_weight}\ndiversity_loss_weight: {diversity_loss_weight}\nap_lr: {ap_lr}\nap_weight_decay: {ap_weight_decay}\nema_decay: {ema_decay}')

    ap_criterion = AdaptivePatchLoss(
        attn_temperature=attn_temperature,
        top_k_focus=top_k_focus,
        rand_samples=6,
        attn_loss_weight=attn_loss_weight,
        diversity_loss_weight=diversity_loss_weight
    )
    vit_criterion = nn.CrossEntropyLoss()
    criterion = [ap_criterion, vit_criterion]

    def param_filter(module, condition):
        return [p for n, p in module.named_parameters() if condition(n)]

    ap_params = param_filter(model.vit.adaptive_patches, lambda n: 'bias' not in n)
    ap_bias_params = param_filter(model.vit.adaptive_patches, lambda n: 'bias' in n)
    vit_params = param_filter(model, lambda n: 'vit.adaptive_patches' not in n and 'bias' not in n)
    vit_bias_params = param_filter(model, lambda n: 'vit.adaptive_patches' not in n and 'bias' in n)
    optimizer = torch.optim.AdamW([
        {'params': ap_params, 'lr': ap_lr, 'weight_decay': ap_weight_decay},
        {'params': ap_bias_params, 'lr': ap_lr, 'weight_decay': 0.0},
        {'params': vit_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': vit_bias_params, 'lr': lr, 'weight_decay': 0.0}
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
            vit_train_loss, ap_train_loss = train(
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
                ema_decay,
                mixup_fn,
                device
            )
            vit_test_loss, ap_test_loss, accuracy = evaluate(
                model,
                testloader,
                criterion,
                device
            )

            trial.report(accuracy, epoch)

            print(f"Epoch: {epoch + 1}/{epochs} | ViT Train Loss: {vit_train_loss:.4f} | ViT Test Loss: {vit_test_loss:.4f} | Accuracy: {accuracy*100:.2f}% | AP Train Loss: {ap_train_loss:.4f}, AP Test Loss: {ap_test_loss:.4f}, | LR: {optimizer.param_groups[0]['lr']:.6f}")

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

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="APViT_Cifar10_AP_HPARAMS", storage="sqlite:///APViT_Cifar10_AP_HPARAMS.db", load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print("Best hyperparameters: ", trial.params)

    with open("best_trial.yaml", "w") as file:
        yaml.dump(trial.params, file)

if __name__ == "__main__": main()
