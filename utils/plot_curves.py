import matplotlib.pyplot as plt
from matplotlib import cm

def read_losses(filename):
    losses = []
    with open(filename, "r") as file:
        for line in file:
            values = line.strip().split(",")
            losses.append({
                "train_loss": float(values[0]),
                "test_loss": float(values[1]),
                "accuracy": float(values[2])
            })
    return losses

def plot_losses(epochs, losses, labels, colors, linestyles):
    for loss_curve, label, color, linestyle in zip(losses, labels, colors, linestyles):
        plt.plot(epochs, loss_curve, label=label, color=color, linestyle=linestyle)
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Train | Test Loss Difference")
    plt.title("Train | Test Loss Difference W vs W/ Pos Embed")
    plt.legend()
    plt.show()

def main():
    files = [
    ]

    labels = [
    ]

    linestyles = [
    ]

    colors = [
    ]

    losses = [read_losses(file) for file in files]
    epochs = range(1, len(losses[0]) + 1)

    # Loss curves
    plot_data = [[loss["test_loss"] for loss in loss_group] for loss_group in losses]

    # Train - Test loss curves
    plot_data = [
        [loss["train_loss"] - loss["test_loss"] for loss in loss_group]
        for loss_group in losses
    ]

    plot_losses(epochs, plot_data, labels, colors, linestyles)



if __name__ == "__main__":
    main()
