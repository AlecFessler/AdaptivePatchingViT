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
from modules.APViT2 import ViT
from modules.AdaptivePatching import AdaptivePatching
from modules.InterpolatePosEmbeds import interpolate_pos_embeds
from modules.PerturbTransformParams import perturb_transform_params
from timm.data import Mixup, create_transform

class APViT(nn.Module):
    def __init__(
        self,
        num_patches,
        patch_size,
        hidden_channels,
        embed_dim,
        num_transformer_layers,
        stochastic_depth,
        scaling,
        max_scale,
        rotating
    ):
        super(APViT, self).__init__()
        self.patch_selector = AdaptivePatching(
            in_channels=3,
            hidden_channels=hidden_channels,
            channel_height=32,
            channel_width=32,
            num_patches=num_patches,
            patch_size=patch_size,
            scaling=scaling,
            max_scale=max_scale,
            rotating=rotating
        )
        self.vit = ViT(
            img_size=32,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            attn_heads=4,
            num_transformer_layers=num_transformer_layers,
            stochastic_depth=stochastic_depth
        )

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
            transform_params = model.patch_selector(inputs)
            patches = model.patch_selector.sample_patches(inputs, transform_params)
            patches = patches.view(-1, patches.size(2), patches.size(3), patches.size(4))
            translate_params = transform_params[:, :, :2]
            pos_embeds = model.vit.pos_embeds.squeeze(0)
            interpolated_pos_embeds = interpolate_pos_embeds(
                pos_embeds,
                translate_params
            ).unsqueeze(1)
            outputs = model.vit(patches, interpolated_pos_embeds)
            loss = criterion(outputs, labels).mean()
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
        criterions,
        optimizers,
        schedulers,
        warmup_schedulers,
        warmup_epochs,
        epoch,
        accumulation_steps,
        scalers,
        mixup_fn,
        ema_decay,
        perturbed_patch_sets,
        device
    ):

    ap_crit, vit_crit = criterions
    ap_opt, vit_opt = optimizers
    ap_sched, vit_sched = schedulers
    ap_wsched, vit_wsched = warmup_schedulers
    ap_scaler, vit_scaler = scalers

    model.train()
    running_vit_loss = 0.0
    running_ap_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            images, labels = mixup_fn(images, labels)

            if i % accumulation_steps == 0:
                ap_opt.zero_grad()
                vit_opt.zero_grad()

            with autocast(device_type=device.type):
                transform_params = model.patch_selector(images)
                patches = model.patch_selector.sample_patches(images, transform_params)
                # combine batch and num patches dims for input to ViTs patch embed layer
                patches = patches.view(-1, *patches.shape[2:])

                # generate a set of randomly perturbed transform params and patches
                perturbed_params = []
                perturbed_patches = []
                for _ in range(perturbed_patch_sets):
                    params = perturb_transform_params(transform_params.clone().detach())
                    perturbed_params.append(params)
                    p_patches = model.patch_selector.sample_patches(images, params)
                    p_patches = p_patches.view(-1, *p_patches.shape[2:])
                    perturbed_patches.append(p_patches)

                # combine original and perturbed params and patches for batching
                perturbed_params = torch.cat(perturbed_params, dim=0)
                transform_params = torch.cat([transform_params, perturbed_params], dim=0)
                perturbed_patches = torch.cat(perturbed_patches, dim=0)
                patches = torch.cat([patches, perturbed_patches], dim=0)

                # interpolate pos embeds for each set of transform params
                translate_params = transform_params[:, :, :2]
                pos_embeds = model.vit.pos_embeds.squeeze(0)
                interpolated_pos_embeds = interpolate_pos_embeds(
                    pos_embeds,
                    translate_params
                ).unsqueeze(1)

                # forward pass through the ViT and compute cross entropy without reduction
                outputs = model.vit(patches, interpolated_pos_embeds)
                repeated_labels = torch.cat([labels for _ in range(perturbed_patch_sets + 1)], dim=0)
                losses = vit_crit(outputs, repeated_labels)

                # select the best set of transform params per image based on lowest loss from perturbed sets
                loss_tensor = losses.view(perturbed_patch_sets + 1, images.size(0))
                _, min_indices = torch.min(loss_tensor, dim=0)
                params_tensor = transform_params.view(images.size(0), perturbed_patch_sets + 1, -1, 5)
                min_params = params_tensor[torch.arange(images.size(0)), min_indices, :, :]
                orig_params = params_tensor[:, 0, :, :]

                # compute the mean cross entropy on the original set of transform params for ViT
                # compute MSE between original and best set of transform params for AP
                vit_loss = loss_tensor[0].mean()
                ap_loss = ap_crit(min_params, orig_params) * 1000

            scaled_ap_loss = ap_loss / accumulation_steps
            ap_scaler.scale(scaled_ap_loss).backward(retain_graph=True)

            scaled_vit_loss = vit_loss / accumulation_steps
            vit_scaler.scale(scaled_vit_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                ap_weights = {k: v.clone().detach() for k, v in model.patch_selector.state_dict().items()}
                ap_scaler.step(ap_opt)
                ap_scaler.update()
                for name, param in model.patch_selector.named_parameters():
                    param.data = ema_decay * param.data + (1 - ema_decay) * ap_weights[name]

                vit_scaler.step(vit_opt)
                vit_scaler.update()

            tepoch.set_postfix(loss=running_vit_loss / (i + 1))
            running_vit_loss += vit_loss.item()
            running_ap_loss += ap_loss.item()

    if epoch < warmup_epochs:
        ap_wsched.step()
        vit_wsched.step()
    else:
        ap_sched.step()
        vit_sched.step()

    return running_vit_loss / len(train_loader), running_ap_loss / len(train_loader)

def main():
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
    ap_lr = config.get("ap_lr", 0.0002)
    ema_decay = config.get("ema_decay", 0.9)
    ap_weight_decay = config.get("ap_weight_decay", 0.009)
    perturbed_patch_sets = config.get("perturbed_patch_sets", 4)

    trainloader, testloader = get_dataloaders(
        batch_size,
        num_workers=4,
        augment_magnitude=augment_magnitude,
        re_prob=re_prob
    )

    model = APViT(
        num_patches=16,
        patch_size=8,
        hidden_channels=hidden_channels,
        embed_dim=attn_embed_dim,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth,
        scaling=None,
        max_scale=0.4,
        rotating=False
    ).to(device)

    ap_crit = nn.MSELoss()
    vit_crit = nn.CrossEntropyLoss(reduction='none')
    criterions = (ap_crit, vit_crit)

    ap_opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
    ], lr=ap_lr, weight_decay=ap_weight_decay)

    vit_opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n]},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
    ], lr=lr, weight_decay=weight_decay)
    optimizers = (ap_opt, vit_opt)

    ap_sched = CosineAnnealingLR(
        ap_opt,
        T_max=epochs-warmup_epochs,
        eta_min=eta_min
    )
    vit_sched = CosineAnnealingLR(
        vit_opt,
        T_max=epochs-warmup_epochs,
        eta_min=eta_min
    )
    schedulers = (ap_sched, vit_sched)

    ap_wsched = LambdaLR(
        ap_opt,
        lr_lambda=lambda epoch: epoch / warmup_epochs
    )
    vit_wsched = LambdaLR(
        vit_opt,
        lr_lambda=lambda epoch: epoch / warmup_epochs
    )
    warmup_schedulers = (ap_wsched, vit_wsched)

    scalers = (GradScaler(), GradScaler())

    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mixup_prob,
        switch_prob=mixup_switch_prob,
        mode='batch', label_smoothing=label_smoothing,
        num_classes=10
    )

    best_weights = None
    best_accuracy = 0.0

    for epoch in range(epochs):
        vit_train_loss, ap_train_loss = train(
            model,
            trainloader,
            criterions,
            optimizers,
            schedulers,
            warmup_schedulers,
            warmup_epochs,
            epoch,
            accumulation_steps,
            scalers,
            mixup_fn,
            ema_decay,
            perturbed_patch_sets,
            device
        )
        vit_test_loss, accuracy = evaluate(
            model,
            testloader,
            criterions[1],
            device
        )

        print(f"Epoch: {epoch + 1}/{epochs} | ViT Train Loss: {vit_train_loss:.4f} | ViT Test Loss: {vit_test_loss:.4f} | Accuracy: {accuracy*100:.2f}% | AP Train Loss: {ap_train_loss:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}

        #with open(f"experiments/training_data/apvit_cifar10_{num_patches}.txt", "a") as file:
            #file.write(f"{vit_train_loss},{vit_test_loss},{accuracy}{ap_train_loss},{ap_test_loss}\n")

        print(f"Best Accuracy: {best_accuracy:.4f}")
        torch.save(best_weights, f"models/apvit_cifar10_16.pth")

if __name__ == "__main__":
    main()
