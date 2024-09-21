import matplotlib.pyplot as plt
from matplotlib import cm

def read_losses(filename):
    losses = []
    with open(filename, "r") as file:
        for line in file:
            losses.append(float(line.strip()))
    return losses

# File paths
dps_vit_files = [
    "../training_data/dps_vit_test_loss_8_patches.txt",
    "../training_data/dps_vit_test_loss_10_patches.txt",
    "../training_data/dps_vit_test_loss_12_patches.txt",
    "../training_data/dps_vit_test_loss_14_patches.txt",
    "../training_data/dps_vit_test_loss_16_patches.txt",
]

standard_vit_file = "../training_data/std_vit_test_loss.txt"

# Read the losses
dps_vit_losses = [read_losses(file) for file in dps_vit_files]
standard_vit_losses = read_losses(standard_vit_file)

# Generate epochs range (assuming all models have the same number of epochs)
epochs = range(1, len(standard_vit_losses) + 1)

# Create a color map for the DpsViT models (light blue to dark blue)
colors = cm.Blues([0.2, 0.4, 0.6, 0.8, 1.0])

# Plot DpsViT models with varying shades of blue
for i, (losses, color) in enumerate(zip(dps_vit_losses, colors)):
    plt.plot(epochs, losses, label=f"DpsViT {8 + i * 2} patches", color=color)

# Plot the StandardViT model in red
plt.plot(epochs, standard_vit_losses, label="StandardViT", color="red")

# Customize the plot
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss per Epoch")
plt.legend()

# Show the plot
plt.show()
