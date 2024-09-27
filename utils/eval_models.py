import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.DpsViT import DpsViT
from modules.StandardViT import StandardViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_loader(batch_size, num_workers):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
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
    return testloader

def remove_prefix_from_state_dict(state_dict, prefix='vit.'):
    return {k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)}

def load_model_weights(model, weights_path, device, prefix='vit.'):
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = remove_prefix_from_state_dict(state_dict, prefix)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    elapsed_time = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total * 100
    elapsed_time = elapsed_time / 1000

    return accuracy, elapsed_time

def load_model(model_path):
    if "std_vit" in model_path:
        model = StandardViT().to(device)
    else:
        model = DpsViT().to(device)
    return model

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = "models/dps_vit_cifar10_8_patches.pth"

    batch_size = 256
    num_workers = 4
    test_loader = get_test_loader(batch_size, num_workers)

    model = load_model(model_path)
    model = load_model_weights(model, model_path, device)

    accuracy, time_taken = evaluate_model(model, test_loader, device)
    print(f"Model: {model_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time Taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
