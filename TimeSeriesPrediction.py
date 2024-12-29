import os
import time
import datetime

from  tools.Config import *
from  tools.PreProcessing import check_for_saved_params
from  tools.PreProcessing import preprocess_data
from  tools.PreProcessing import split_data
from  tools.Training import load_and_train_model
from  tools.Evaluation import evaluate_model
from  tools.FuturePrediction import future_prediction

root_address = os.path.dirname(os.path.abspath(__file__))

def main():
    """ main function to execute the time series modeling process."""
    # Setup output directories and logging
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_save_dir = os.path.join(root_address, 'results', f'run_{date_time}')
    os.makedirs(results_save_dir, exist_ok=True)

    log_file_path = os.path.join(results_save_dir, 'process.log')
    logger = setup_logger(log_file_path)
    log_system_info(logger)

    logger.info("Initializing the time series modeling workflow...")

    # Define default parameters
    default_params = {
        'file_path': os.path.join(root_address, 'dataset', 'data_xauusd_short.csv'),
        'window_size': 720,
        'normalization_method': 'standard',
        'imputation_method': 'knn',
        'imputation_neighbors': 5,
        'outlier_removal': False,
        'outlier_method': 'zscore',
        'feature_extraction': True,
        'time_domain_features': True,
        'frequency_domain_features': True,
        'wavelet_features': True,
        'use_original': True,
        'use_difference': True,
        'epochs': 24,
        'batch_size': 256,
        'patience': 10,
        'horizon': 150,
        'n_splits': 5,
        'alpha_sections': 10,
        'is_plot': True,
        'optimizer': 'adam',
        'learning_rate': 0.0001,
        'loss_function': 'mse',
        'retrain': True,
        'model_path': "./results/run_20241229_235030/best_model_full_data.keras",
        'params_path': "./results/run_20241229_235030/Model_params.pkl",
        'split_method': 'sectional',
        'save_model_option': 'best_train_loss',  # or 'latest_epoch'
        'metrics': ['mae', 'mape'],
        'model_save_dir': os.path.join(root_address, 'models'),
        'results_save_dir': results_save_dir,
        'cnn_filters': 128,
        'kernel_size': 3,
        'dropout_rate': 0.2,
        'num_heads': 8,
        'key_dim': 64,
        'dff': 256,
        'num_cnn_layers': 4,
        'num_attention_layers': 2,
        'dense_units': [256, 128],
        'dpi': 80,
    }

    params = default_params.copy()
    start_time = time.time()

    # Attempt to load external params if params_path is provided
    params = check_for_saved_params(params,logger)

    # Preprocessing data
    logger.info("Preprocessing data...")
    X, y, initial_values, data, original_data, scaler = preprocess_data(params, logger)

    # Determine input and target shapes
    input_shape = (X.shape[1],) if len(X.shape) == 2 else X.shape[1:]
    target_shape = (y.shape[1],)

    # Split data into training and testing sets
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test, train_indices, test_indices, initial_values_train, initial_values_test = split_data(
        X, y, initial_values, params, logger
    )

    # Ensure results_save_dir is consistent in params
    model_results_dir = params['results_save_dir']

    # Load model (if provided) or build a new one and train
    logger.info("Loading/Building and training the model...")
    model, history = load_and_train_model(input_shape, target_shape, X_train, y_train, X_test, y_test, params, logger)

    # Model evaluation
    logger.info("Evaluating the model...")
    if params.get('use_difference', False):
        results, model_file, params_file = evaluate_model(
            model, X_train, y_train, X_test, y_test, 'Model', params, scaler,
            initial_values_train, initial_values_test, train_indices, test_indices,
            original_data, model_results_dir
        )
    else:
        results, model_file, params_file = evaluate_model(
            model, X_train, y_train, X_test, y_test, 'Model', params, scaler,
            train_indices=train_indices, test_indices=test_indices,
            original_data=original_data, results_dir=model_results_dir
        )
    logger.info(f"Model Evaluation Results:\n{results}")

    # Future prediction
    logger.info("Performing future predictions...")
    predictions = future_prediction(
        model, data, original_data, params, scaler, results_dir=model_results_dir
    )

    # Time tracking
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
