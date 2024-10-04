import optuna

def visualize_grid_results():
    study = optuna.load_study(
        study_name="Ablation_Study",
        storage="sqlite:///ablation_study.db"
    )

    df = study.trials_dataframe()

    print(df[['number', 'value', 'params_scaling', 'params_rotating', 'params_ema', 'params_ap_loss']])

    df.to_csv("grid_search_results.csv")

if __name__ == "__main__":
    visualize_grid_results()
