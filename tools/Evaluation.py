
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE with handling for zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    # Using updated style that works with newer versions
    plt.style.use('grayscale') # Base style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.right': True,
        'axes.spines.top': True
    })

def plot_prediction_comparison(ax, indices, actual, predicted, model_name):
    """Create publication-quality prediction comparison plot."""
    ax.plot(indices, actual, label='Actual', color='#1f77b4', linewidth=1)
    ax.plot(indices, predicted, label='Predicted', color='#ff7f0e', linewidth=1, linestyle='--')
    ax.set_title(f'Time Series Prediction Results: {model_name}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Load Consumption')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.minorticks_on()

def plot_residuals(ax, indices, actual, predicted, model_name):
    """Create publication-quality residuals plot."""
    residuals = actual - predicted
    ax.plot(indices, residuals, color='#2ecc71', linewidth=1)
    ax.set_title('Residuals Analysis')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Residual Value')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.minorticks_on()

def plot_error_distribution(ax, actual, predicted, model_name):
    """Create publication-quality error distribution plot."""
    residuals = actual - predicted
    sns.histplot(residuals, kde=True, ax=ax, color='#3498db')
    ax.set_title('Error Distribution')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')

def create_scatter_plot(actual, predicted, model_name, results_dir, params):
    """Create publication-quality scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5, color='#9b59b6')

    # Add diagonal line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    scatter_path = os.path.join(results_dir, f"{model_name}_scatter")
    save_figure(plt, scatter_path, params)
    plt.close()

def save_figure(plt, save_path, params):
    """Save figure in specified format with error handling."""
    try:
        # Get format from params or default to 'png'
        img_format = params.get('image_format', 'png').lower()
        # Remove any leading dot if present
        img_format = img_format.lstrip('.')

        # Create the full path with correct extension
        base_path = os.path.splitext(save_path)[0]
        full_path = f"{base_path}.{img_format}"

        plt.savefig(full_path, format=img_format, bbox_inches='tight', dpi=300)
        return full_path
    except Exception as e:
        logging.error(f"Error saving figure: {str(e)}")
        # Fallback to PNG if specified format fails
        fallback_path = f"{base_path}.png"
        plt.savefig(fallback_path, format='png', bbox_inches='tight', dpi=300)
        return fallback_path

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, params, scaler=None,
                   initial_values_train=None, initial_values_test=None, train_indices=None, test_indices=None,
                   original_data=None, results_dir=None):
    """Evaluate the model and plot results."""
    logger = logging.getLogger('TimeSeriesModel')
    logger.info(f"Evaluating model: {model_name}")

    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Concatenate all for convenience
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    y_all_pred = np.concatenate((y_train_pred, y_test_pred), axis=0)

    # Handle differencing if used
    use_difference = params.get('use_difference', False)
    if use_difference:
        if initial_values_train is None or initial_values_test is None:
            raise ValueError("initial_values_train and initial_values_test are required for differencing")
        initial_values_all = np.concatenate((initial_values_train, initial_values_test), axis=0)
    else:
        initial_values_all = None

    # Inverse scaling if scaler is provided
    if scaler is not None:
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_all = scaler.inverse_transform(y_all)
        y_all_pred = scaler.inverse_transform(y_all_pred)

        if use_difference and initial_values_train is not None and initial_values_test is not None:
            initial_values_train = scaler.inverse_transform(initial_values_train)
            initial_values_test = scaler.inverse_transform(initial_values_test)
            initial_values_all = np.concatenate((initial_values_train, initial_values_test), axis=0)

    # If differencing was used, reconstruct the original scale
    if use_difference:
        y_train_de_diff = initial_values_train + y_train
        y_train_pred_de_diff = initial_values_train + y_train_pred
        y_test_de_diff = initial_values_test + y_test
        y_test_pred_de_diff = initial_values_test + y_test_pred
        y_all_de_diff = initial_values_all + y_all
        y_all_pred_de_diff = initial_values_all + y_all_pred
    else:
        y_train_de_diff = y_train
        y_train_pred_de_diff = y_train_pred
        y_test_de_diff = y_test
        y_test_pred_de_diff = y_test_pred
        y_all_de_diff = y_all
        y_all_pred_de_diff = y_all_pred

    # Define evaluation metrics
    metrics = {
        'MSE': mean_squared_error,
        'RMSE': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'R2_Score': r2_score
    }

    def calculate_advanced_metrics(actual, predicted):
        residuals = actual - predicted
        return {
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals),
            'Durbin_Watson': durbin_watson(residuals)
        }

    # Calculate metrics
    results = {}
    for name, func in metrics.items():
        train_metric = func(y_train_de_diff, y_train_pred_de_diff)
        test_metric = func(y_test_de_diff, y_test_pred_de_diff)
        results[name] = {'train': train_metric, 'test': test_metric}

    # Calculate advanced metrics
    advanced_train = calculate_advanced_metrics(y_train_de_diff.flatten(), y_train_pred_de_diff.flatten())
    advanced_test = calculate_advanced_metrics(y_test_de_diff.flatten(), y_test_pred_de_diff.flatten())
    for key, value in advanced_train.items():
        results[key] = {'train': value, 'test': advanced_test[key]}

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': list(results.keys()),
        'Train': [results[k]['train'] for k in results.keys()],
        'Test': [results[k]['test'] for k in results.keys()]
    })
    metrics_df.to_csv(os.path.join(results_dir, f"{model_name}_metrics.csv"), index=False)

    # Save model and parameters
    model_file = os.path.join(results_dir, f"best_model_{model_name}.h5")
    model.save(model_file)
    params_file = os.path.join(results_dir, f"{model_name}_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)

    if params.get('is_plot', False):
        try:
            set_publication_style()
        except Exception as e:
            logger.warning(f"Could not set publication style: {str(e)}. Using default style.")
            plt.style.use('grayscale')

        # Prepare sorted indices and data for plotting
        total_indices = np.concatenate((train_indices, test_indices))
        total_indices = np.array(total_indices).flatten()
        y_all_de_diff_flat = y_all_de_diff.flatten()
        y_all_pred_de_diff_flat = y_all_pred_de_diff.flatten()
        sorted_order = np.argsort(total_indices)
        total_indices_sorted = total_indices[sorted_order]
        y_all_sorted = y_all_de_diff_flat[sorted_order]
        y_all_pred_sorted = y_all_pred_de_diff_flat[sorted_order]

        # Create main analysis plots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)

        # Plot 1: Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, :])
        plot_prediction_comparison(ax1, total_indices_sorted, y_all_sorted, y_all_pred_sorted, model_name)

        # Plot 2: Residuals
        ax2 = fig.add_subplot(gs[1, 0])
        plot_residuals(ax2, total_indices_sorted, y_all_de_diff_flat, y_all_pred_de_diff_flat, model_name)

        # Plot 3: Error Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        plot_error_distribution(ax3, y_all_de_diff_flat, y_all_pred_de_diff_flat, model_name)

        plt.tight_layout()

        # Save with specified format
        analysis_path = os.path.join(results_dir, f"{model_name}_analysis")
        save_figure(plt, analysis_path, params)

        if params.get('show_plots', False):
            plt.show()
        plt.close()

        # Create and save additional plots
        create_scatter_plot(y_all_de_diff_flat, y_all_pred_de_diff_flat, model_name, results_dir, params)
        create_train_test_comparison(train_indices, test_indices,
                                     y_train_de_diff, y_train_pred_de_diff,
                                     y_test_de_diff, y_test_pred_de_diff,
                                     model_name, results_dir, params)

    return results, model_file, params_file


def create_train_test_comparison(train_indices, test_indices, y_train, y_train_pred,
                                 y_test, y_test_pred, model_name, results_dir, params):
    """Create publication-quality train-test comparison plot."""
    plt.figure(figsize=(15, 6))

    # Plot training data
    plt.plot(train_indices, y_train, color='#2980b9', label='Train Actual')
    plt.plot(train_indices, y_train_pred, '--', color='#3498db', label='Train Predicted')

    # Plot testing data
    plt.plot(test_indices, y_test, color='#c0392b', label='Test Actual')
    plt.plot(test_indices, y_test_pred, '--', color='#e74c3c', label='Test Predicted')

    plt.title('Training and Testing Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Load Consumption')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    train_test_path = os.path.join(results_dir, f"{model_name}_train_test")
    save_figure(plt, train_test_path, params)
    plt.close()

