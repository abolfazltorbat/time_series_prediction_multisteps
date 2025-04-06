import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
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

def set_publication_style(params):
    """Set matplotlib parameters for publication-quality figures."""
    plt.style.use('grayscale')  # Base style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': params['dpi'],
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
        img_format = img_format.lstrip('.')  # Remove any leading dot

        # Create the full path with correct extension
        base_path = os.path.splitext(save_path)[0]
        full_path = f"{base_path}.{img_format}"

        plt.savefig(full_path, format=img_format, bbox_inches='tight', dpi=params['dpi'])
        # Save the figure as an interactive HTML file
        html_path = f"{base_path}.html"
        mpld3.save_html(plt.gcf(), html_path)

        return full_path
    except Exception as e:
        base_path = os.path.splitext(save_path)[0]
        logging.error(f"Error saving figure: {str(e)}")
        fallback_path = f"{base_path}.png"
        plt.savefig(fallback_path, format='png', bbox_inches='tight', dpi=params['dpi'])
        # Save the fallback figure as an interactive HTML file
        html_fallback_path = f"{base_path}.html"
        mpld3.save_html(plt.gcf(), html_fallback_path)
        return fallback_path

def get_continuous_segments(indices):
    """Get continuous segments from indices."""
    segments = []
    sorted_indices = sorted(indices)
    if not sorted_indices:
        return segments
    start = sorted_indices[0]
    prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))
    return segments

def create_train_test_comparison(train_indices, test_indices, y_train, y_train_pred,
                                 y_test, y_test_pred, model_name, results_dir, params):
    """
    Create publication-quality train-test comparison plot with proper segmentation.
    """
    plt.style.use('grayscale')
    fig, ax = plt.subplots(figsize=(15, 6), dpi=params['dpi'])

    # Color scheme
    colors = {
        'train_actual': '#2980b9',  # Dark blue
        'train_pred': '#3498db',  # Light blue
        'test_actual': '#c0392b',  # Dark red
        'test_pred': '#e74c3c'  # Light red
    }

    # Get continuous segments for train and test data
    train_segments = get_continuous_segments(train_indices)
    test_segments = get_continuous_segments(test_indices)

    # Plot training data segments
    train_data_idx = 0
    for start, end in train_segments:
        segment_length = end - start + 1
        ax.plot(
            range(start, end + 1),
            y_train[train_data_idx:train_data_idx + segment_length],
            color=colors['train_actual'],
            label='Train Actual' if start == train_segments[0][0] else "",
            linewidth=1.5
        )
        ax.plot(
            range(start, end + 1),
            y_train_pred[train_data_idx:train_data_idx + segment_length],
            '--',
            color=colors['train_pred'],
            label='Train Predicted' if start == train_segments[0][0] else "",
            linewidth=1.5
        )
        train_data_idx += segment_length

    # Plot testing data segments
    test_data_idx = 0
    for start, end in test_segments:
        segment_length = end - start + 1
        ax.plot(
            range(start, end + 1),
            y_test[test_data_idx:test_data_idx + segment_length],
            color=colors['test_actual'],
            label='Test Actual' if start == test_segments[0][0] else "",
            linewidth=1.5
        )
        ax.plot(
            range(start, end + 1),
            y_test_pred[test_data_idx:test_data_idx + segment_length],
            '--',
            color=colors['test_pred'],
            label='Test Predicted' if start == test_segments[0][0] else "",
            linewidth=1.5
        )
        test_data_idx += segment_length

    ax.set_title('Training and Testing Predictions',
                 fontsize=14,
                 fontweight='bold',
                 pad=20)

    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Load Consumption', fontsize=12, fontweight='bold')
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.minorticks_on()

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10, ncol=1)

    # Add split ratio information
    split_ratio = len(train_indices) / (len(train_indices) + len(test_indices))
    text_str = f'Split Ratio (Train/Total): {split_ratio:.2%}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props)

    plt.tight_layout()

    train_test_path = os.path.join(results_dir, f"{model_name}_train_test")
    if params.get('save_plot', True):
        plt.savefig(f"{train_test_path}.png",
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0.1,
                    metadata={'Creator': 'Train-Test Comparison Plot'})
        # Save interactive HTML version if mpld3 is available
        try:
            mpld3.save_html(fig, f"{train_test_path}.html")
        except ImportError:
            pass

    if params.get('show_plot', True):
        plt.show()

    plt.close()

# ---------------------------------------------------------------------------------
#   MULTI-STEP FORECASTING + BACKTESTING HELPERS
# ---------------------------------------------------------------------------------

def multi_step_forecast(model, X_data, n_steps,params,is_use_backtest=False):
    """
    Example multi-step forecast function for demonstration.

    - X_data: your input features, shape could be [samples, timesteps, features].
    - n_steps: how many steps ahead to predict for each sample.

    This function should return an array of predictions shaped
    appropriately, e.g. [samples, n_steps].
    """
    back_test_folder = 'back_test'
    output_folder = params['results_save_dir'] + '\\' + back_test_folder

    # Check if folder exists, create if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    predictions = []
    for i in range(len(X_data)):
        # Extract the single sample
        x_input = X_data[i].reshape(1, X_data.shape[1], 1)  # Assuming a 3D input for RNN
        # Predict n_steps ahead (assuming your model can handle it)
        y_pred = model.predict(x_input, verbose=0)  # shape should be (1, n_steps)
        predictions.append(y_pred[0])  # Append the 1D array of predictions

        if is_use_backtest:
            from tools.PlotPrediction import plot_future_prediction
            plot_future_prediction( X_data[i].reshape(-1,1)[120:],
                                   y_pred.reshape(-1,1),
                                   params,
                                   output_folder,
                                   is_in_future_prediction=False,
                                   count = str(i)+'_',
                                   is_plot = False,
                                   date_time=None)

        # Calculate the progress bar
        filled_length = int(40 * (i + 1) / len(X_data))
        bar = '█' * filled_length + '░' * (40 - filled_length)
        print(f"\r[{bar}]  | multi step forecast, Step: {i + 1}/{len(X_data)}", end='')


    return np.array(predictions)  # shape = (samples, n_steps)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

def backtest_on_test_data(model, X_test, y_test, n_steps, scaler, results_dir, model_name,
                          output_mode='plot_and_save', plot_style='dark',
                          date_index=None):
    """
    Enhanced backtesting function with sophisticated plotting and flexible save options.

    Parameters:
    -----------
    model : keras.Model
        Trained model for prediction
    X_test : numpy.array
        Test features
    y_test : numpy.array
        Test targets
    n_steps : int
        Number of steps to predict
    scaler : sklearn.preprocessing
        Scaler used for data normalization
    results_dir : str
        Directory to save results
    model_name : str
        Name of the model for saving files
    output_mode : str, optional (default='plot_and_save')
        One of ['plot_and_save', 'save_only', 'none']
    plot_style : str, optional (default='dark')
        Style for plots ('dark', 'light', 'paper')
    date_index : pandas.DatetimeIndex, optional
        Date index for plotting time series

    Returns:
    --------
    dict
        Dictionary containing all prediction results and error metrics
    """
    # Set up logging
    logger = logging.getLogger('TimeSeriesBacktest')

    # Create results directory if it doesn't exist
    backtest_dir = os.path.join(results_dir, 'back_testing')
    os.makedirs(backtest_dir, exist_ok=True)

    # Initialize containers for storing results
    all_predictions = []
    all_real_values = []
    errors_records = []

    num_samples = len(X_test)
    i = 0
    step_count = 0

    # Set plot style
    if plot_style == 'dark':
        plt.style.use('dark_background')
        line_colors = ['#00ff00', '#ff0000', '#0000ff']
    elif plot_style == 'paper':
        plt.style.use('seaborn-paper')
        line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    else:
        plt.style.use('seaborn')
        line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    while i < num_samples:
        # 1) Generate prediction
        x_input = X_test[i].reshape(1, X_test.shape[1], 1)
        y_pred = model.predict(x_input, verbose=0)
        y_pred = y_pred[0]

        # 2) Get actual values
        n_steps = len(y_pred)
        real_end = min(i + n_steps, num_samples)
        y_real = y_test[i:real_end]

        # 3) Inverse transform if scaler was used
        if scaler is not None:
            y_real = scaler.inverse_transform(y_real.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Store predictions and real values
        all_predictions.append(y_pred)
        all_real_values.append(y_real)

        # 4) Compute metrics
        overlap = len(y_real)
        y_pred_overlap = y_pred[:overlap]

        residuals = y_real - y_pred_overlap
        abs_errors = np.abs(residuals)

        metrics = {
            'Step_Index': step_count,
            'Data_Index': i,
            'Min_Abs_Error': np.min(abs_errors),
            'Max_Abs_Error': np.max(abs_errors),
            'Mean_Abs_Error': np.mean(abs_errors),
            'RMSE': np.sqrt(np.mean(residuals ** 2)),
            'Peak_Real': np.max(y_real),
            'Peak_Pred': np.max(y_pred_overlap),
            'Peak_Diff': np.abs(np.max(y_real) - np.max(y_pred_overlap)),
            'Min_Real': np.min(y_real),
            'Min_Pred': np.min(y_pred_overlap),
            'Min_Diff': np.abs(np.min(y_real) - np.min(y_pred_overlap))
        }
        errors_records.append(metrics)

        # Log progress
        logger.info(f"Backtest Step {step_count}: {metrics}")

        if output_mode != 'none' and step_count % 10 == 0:
            # Create sophisticated plot for current window
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

            # Plot actual vs predicted
            if date_index is not None:
                x_axis = date_index[i:i + len(y_real)]
            else:
                x_axis = np.arange(i, i + len(y_real))

            ax1.plot(x_axis, y_real, label='Actual', color=line_colors[0], linewidth=2)
            ax1.plot(x_axis, y_pred[:len(y_real)], '--', label='Predicted',
                     color=line_colors[1], linewidth=2)

            # Add confidence bands (example: ±2*std of residuals)
            std_residuals = np.std(residuals)
            ax1.fill_between(x_axis,
                             y_pred[:len(y_real)] - 2 * std_residuals,
                             y_pred[:len(y_real)] + 2 * std_residuals,
                             color=line_colors[1], alpha=0.2)

            # Plot residuals
            ax2.bar(x_axis, residuals, color=line_colors[2], alpha=0.6)
            ax2.axhline(y=0, color='white' if plot_style == 'dark' else 'black',
                        linestyle='-', linewidth=0.5)

            # Customize plots
            ax1.set_title(f'Backtest Window {step_count}: Actual vs Predicted',
                          fontsize=12, pad=10)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            ax2.set_title('Residuals', fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot if requested
            if output_mode in ['plot_and_save', 'save_only']:
                plot_path = os.path.join(backtest_dir,
                                         f"{model_name}_backtest_window_{step_count}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')

            if output_mode == 'plot_and_save':
                plt.show()
            plt.close()

        i += n_steps
        step_count += 1

    # Save error metrics
    errors_df = pd.DataFrame(errors_records)
    metrics_path = os.path.join(backtest_dir, f"{model_name}_backtest_metrics.csv")
    errors_df.to_csv(metrics_path, index=False)

    # Calculate and save aggregate metrics
    aggregate_metrics = {
        'Overall_MAE': errors_df['Mean_Abs_Error'].mean(),
        'Overall_RMSE': np.sqrt((errors_df['Mean_Abs_Error'] ** 2).mean()),
        'Max_Peak_Diff': errors_df['Peak_Diff'].max(),
        'Avg_Peak_Diff': errors_df['Peak_Diff'].mean(),
        'Total_Steps': step_count
    }

    # Save aggregate metrics
    agg_metrics_path = os.path.join(backtest_dir, f"{model_name}_aggregate_metrics.csv")
    pd.DataFrame([aggregate_metrics]).to_csv(agg_metrics_path, index=False)

    logger.info(f"Backtest results saved to {backtest_dir}")

    return {
        'predictions': all_predictions,
        'real_values': all_real_values,
        'errors': errors_df,
        'aggregate_metrics': aggregate_metrics
    }

# ---------------------------------------------------------------------------------
#   MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------------
def evaluate_model(model,
                   X_train, y_train,
                   X_test, y_test,
                   model_name, params,
                   scaler=None,
                   initial_values_train=None,
                   initial_values_test=None,
                   train_indices=None,
                   test_indices=None,
                   original_data=None,
                   results_dir=None):
    """
    Evaluate the model and plot results.
    Now supports 'recursive' (one-step) vs. 'multi_steps' approach
    via params['prediction_approach'].
    """
    logger = logging.getLogger('TimeSeriesModel')
    logger.info(f"Evaluating model: {model_name}")

    # --------------------------------------------------------------------
    # 1) Check if we want 'recursive' (one-step) or 'multi_steps'
    # --------------------------------------------------------------------
    prediction_approach = params.get('prediction_approach', 'recursive')
    n_steps = params.get('n_steps', 1)  # number of steps for multi-step

    if prediction_approach == 'recursive':
        # ----------------------------------------------------------------
        # (A) RECURSIVE (ONE-STEP) PREDICTION -- EXACTLY AS BEFORE
        # ----------------------------------------------------------------
        y_train_pred = model.predict(X_train, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)

    else:
        # ----------------------------------------------------------------
        # (B) MULTI-STEP PREDICTION
        # ----------------------------------------------------------------
        logger.info("Using MULTI-STEP approach...")

        # For train set
        #   We'll forecast n_steps for each sample. Then you decide how to
        #   compare that with y_train. Here, we do a simple approach: compare
        #   the first step of each forecast with the real data. Or store all n_steps.
        multi_train_preds = multi_step_forecast(model, X_train, n_steps=n_steps,params=params)
        # For simplicity, let's assume we only compare the first-step forecast
        # with the actual y_train. That means multi_train_preds[:, 0] corresponds 
        # to one-step predictions for each sample.
        y_train_pred = multi_train_preds[:, 0].reshape(-1, 1)


        # For test set
        multi_test_preds = multi_step_forecast(model, X_test, n_steps=n_steps,params=params)
        y_test_pred = multi_test_preds[:, 0].reshape(-1, 1)

    # --------------------------------------------------------------------
    # 2) Combine data for normal large-scale evaluation
    # --------------------------------------------------------------------
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

    if not prediction_approach == 'recursive':
        y_all_de_diff =y_all_de_diff[:, 0].reshape(-1, 1)

    y_train_de_diff = np.squeeze(y_train_de_diff)
    y_train_pred_de_diff = np.squeeze(y_train_pred_de_diff)
    y_test_de_diff = np.squeeze(y_test_de_diff)
    y_test_pred_de_diff = np.squeeze(y_test_pred_de_diff)
    # --------------------------------------------------------------------
    # 3) Large-scale evaluation metrics (same as original)
    # --------------------------------------------------------------------
    metrics_dict = {
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

    results = {}
    for name, func in metrics_dict.items():
        if prediction_approach == 'multi_steps':
            train_metric = func(np.squeeze(y_train_de_diff[:, 0].reshape(-1, 1)), y_train_pred_de_diff)
            test_metric = func(np.squeeze(y_test_de_diff[:, 0].reshape(-1, 1)), y_test_pred_de_diff)
        else:
            train_metric = func(y_train_de_diff, y_train_pred_de_diff)
            test_metric = func(y_test_de_diff, y_test_pred_de_diff)

        results[name] = {'train': train_metric, 'test': test_metric}


    # advanced metrics
    if prediction_approach == 'multi_steps':
        y_train_de_diff = np.squeeze(y_train_de_diff[:, 0].reshape(-1, 1))
        y_test_de_diff = np.squeeze(y_test_de_diff[:, 0].reshape(-1, 1))

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
    metrics_csv_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Metrics saved to {metrics_csv_path}")

    # Save model and parameters
    model_file = os.path.join(results_dir, f"best_model_{model_name}.h5")
    model.save(model_file)
    params_file = os.path.join(results_dir, f"{model_name}_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)

    # --------------------------------------------------------------------
    # 4) (Optional) If multi_steps, run a backtest loop
    # --------------------------------------------------------------------
    # if prediction_approach == 'multi_steps':
    #     logger.info("Running backtest on test data for multi-step approach...")
    #     results = backtest_on_test_data(
    #         model=model,
    #         X_test=X_test,
    #         y_test=y_test,
    #         n_steps=n_steps,
    #         scaler=scaler,
    #         results_dir=results_dir,
    #         model_name=model_name
    #     )

    # --------------------------------------------------------------------
    # 5) Plotting
    # --------------------------------------------------------------------
    if params.get('is_plot', False):
        try:
            set_publication_style(params)
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

        analysis_path = os.path.join(results_dir, f"{model_name}_analysis")
        save_figure(plt, analysis_path, params)

        if params.get('is_plot', False):
            plt.show()
        plt.close()

        # Create and save additional plots
        create_scatter_plot(y_all_de_diff_flat, y_all_pred_de_diff_flat, model_name, results_dir, params)

        create_train_test_comparison(train_indices, test_indices,
                                     y_train_de_diff, y_train_pred_de_diff,
                                     y_test_de_diff, y_test_pred_de_diff,
                                     model_name, results_dir, params)

    return results, model_file, params_file
