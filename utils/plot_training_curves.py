import matplotlib.pyplot as plt

def read_losses(filename):
    losses = []
    with open(filename, "r") as file:
        for line in file:
            losses.append(float(line.strip()))
    return losses

dps_vit_losses = read_losses("../training_data/dps_vit_test_loss_bounded_translations.txt")
standard_vit_losses = read_losses("../training_data/std_vit_test_loss.txt")

epochs = range(1, len(dps_vit_losses) + 1)

plt.plot(epochs, dps_vit_losses, label="DpsViT", color="blue")
plt.plot(epochs, standard_vit_losses, label="StandardViT", color="red")

plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss per Epoch")
plt.legend()

plt.show()
