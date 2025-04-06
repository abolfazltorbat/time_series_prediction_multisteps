import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime


def backtest_data(model, X_test, y_test, scaler, results_dir, model_name,
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
