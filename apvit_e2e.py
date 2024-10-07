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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import os
import yaml
from copy import deepcopy
from modules.APViT import APViT
from modules.ValueScheduler import ValueScheduler
from timm.data import Mixup, create_transform
from utils.save_patch_grid import save_patch_grid
from utils.plot_attn_scores import plot_attention_scores

class APViT_E2E(nn.Module):
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
        super(APViT_E2E, self).__init__()
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

def evaluate_analysis(model, test_loader, device, output_dir="./output"):
    model.eval()
    model.vit.setup_hooks()

    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)

    model.vit.remove_hooks()

    selected_patches = model.vit.selected_patches
    translate_params = model.vit.translate_params
    #scale_params = model.vit.scale_params
    #rotate_params = model.vit.rotate_params

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, predicted_labels = outputs.max(1)

    for img_num in range(inputs.size(0)):
        with open(os.path.join(output_dir, f"params_{img_num}.txt"), "w") as file:
            file.write(f"Image Number: {img_num}\n")
            file.write(f"Actual Label: {labels[img_num].item()}\n")
            file.write(f"Predicted Label: {predicted_labels[img_num].item()}\n")
            file.write(f"Translation Params: {translate_params[img_num].cpu().numpy()}\n")
            #file.write(f"Scale Params: {scale_params[img_num].cpu().numpy()}\n")
            #file.write(f"Rotate Params: {rotate_params[img_num].cpu().numpy()}\n")

        resize = transforms.Resize((256, 256))
        resized_img = resize(inputs[img_num].cpu())
        torchvision.utils.save_image(resized_img, os.path.join(output_dir, f"img_{img_num}.png"))

        save_patch_grid(
            patches=selected_patches[img_num],
            translation_params=translate_params[img_num],
            output_path=os.path.join(output_dir, f"grid_{img_num}.png"),
            channels=inputs.size(1),
            patch_size=selected_patches.size(-1),
            resize_dim=(512, 512)
        )

        # Extract attention weights for the specific image in the batch
        #plot_attention_scores(
        #    attn_weights=attn_weights[img_num],
        #    translation_params=translate_params[img_num],
        #    output_path=os.path.join(output_dir, f"attention_summary_{img_num}.png")
        #)

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
        ema_model,
        ema_sched,
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
                raise ValueError("Loss is NaN")

            scaled_vit_loss = vit_loss / accumulation_steps
            scaler.scale(scaled_vit_loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_sched.current_value).add_((1 - ema_sched.current_value) * param.data)

            running_vit_loss += vit_loss.item()
            tepoch.set_postfix(loss=running_vit_loss / (i + 1))

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        scheduler.step()

    ema_sched.step()

    return running_vit_loss / len(train_loader)

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
    lr = config.get("vit_lr", 0.0004)
    eta_min = config.get("vit_lr_min", 0.0001)
    weight_decay = config.get("vit_weight_decay", 0.000015)

    trainloader, testloader = get_dataloaders(
        batch_size,
        num_workers=4,
        augment_magnitude=augment_magnitude,
        re_prob=re_prob
    )

    patches_tests = [16, 14, 12, 10, 8]
    for num_patches in patches_tests:

        model = APViT_E2E(
            num_patches,
            hidden_channels=hidden_channels,
            attn_embed_dim=attn_embed_dim,
            num_transformer_layers=num_transformer_layers,
            stochastic_depth=stochastic_depth,
            pos_embed_size=4,
            scaling=None,
            max_scale=0.4,
            rotating=False
        ).to(device)

        ema_model = deepcopy(model)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'bias' not in n]},
            {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
        ], lr=lr, weight_decay=weight_decay)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs-warmup_epochs,
            eta_min=eta_min
        )

        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: epoch / warmup_epochs
        )

        ema_sched = ValueScheduler(
            start=0.996,
            end=1.0,
            steps=epochs,
            cosine=True
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

        best_weights = None
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
                ema_model,
                ema_sched,
                device
            )
            vit_test_loss, accuracy = evaluate(
                ema_model,
                testloader,
                criterion,
                device
            )

            print(f"Epoch: {epoch + 1}/{epochs} | ViT Train Loss: {vit_train_loss:.4f} | ViT Test Loss: {vit_test_loss:.4f} | Accuracy: {accuracy*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = ema_model.state_dict()

            with open(f"experiments/training_data/apvit_e2e_{num_patches}.txt", "a") as file:
                file.write(f"{vit_train_loss},{vit_test_loss},{accuracy}\n")

        print(f"Best Accuracy: {best_accuracy:.4f}")
        torch.save(best_weights, f"models/apvit_e2e_{num_patches}.pth")

def eval_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("hparams_config.yaml")

    num_patches = 16
    hidden_channels = config.get("hidden_channels", 16)
    attn_embed_dim = config.get("attn_embed_dim", 256)
    num_transformer_layers = config.get("num_transformer_layers", 8)
    stochastic_depth = config.get("stochastic_depth", 0.15)

    model = APViT_E2E(
        num_patches,
        hidden_channels=hidden_channels,
        attn_embed_dim=attn_embed_dim,
        num_transformer_layers=num_transformer_layers,
        stochastic_depth=stochastic_depth,
        pos_embed_size=4,
        scaling=None,
        max_scale=0.3,
        rotating=False
    ).to(device)

    pretrained_weights = torch.load("models/apvit_cifar10_16.pth", map_location=device)
    model.load_state_dict(pretrained_weights)

    _, testloader = get_dataloaders(batch_size=10, num_workers=2)

    output_dir = "./evaluation_output"
    evaluate_analysis(model, testloader, device, output_dir=output_dir)

    print(f"Evaluation complete. Results saved in {output_dir}")

#if __name__ == "__main__": eval_main()

if __name__ == "__main__": main()
