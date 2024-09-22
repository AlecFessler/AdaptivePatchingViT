import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_contour

def visualize_study():
    study = optuna.load_study(study_name="DpsViT_Cifar10", storage="sqlite:///DpsViT_Cifar10.db")

    print(study.best_params)

    opt_history = plot_optimization_history(study)
    opt_history.show()

    param_importances = plot_param_importances(study)
    param_importances.show()

    parallel_coordinate = plot_parallel_coordinate(study)
    parallel_coordinate.show()

    contour_plot = plot_contour(study)
    contour_plot.show()

if __name__ == "__main__":
    visualize_study()
