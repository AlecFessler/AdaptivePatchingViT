import seaborn as sns
import matplotlib.pyplot as plt
import optuna

def plot_heatmap():
    study = optuna.load_study(
        study_name="Ablation_Study",
        storage="sqlite:///ablation_study.db"
    )

    df = study.trials_dataframe()

    heatmap_data = df.pivot_table(
        index='params_scaling',
        columns='params_rotating',
        values='value'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
    plt.title("Performance of Different Hyperparameter Combinations")
    plt.xlabel("Rotating")
    plt.ylabel("Scaling")
    plt.show()

if __name__ == "__main__":
    plot_heatmap()
