import csv
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.fft import fft
from scipy.stats import skew, kurtosis, iqr, entropy
from tqdm import tqdm


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import tensorflow as tf




from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, Dropout, Input, Flatten, Conv1D, LayerNormalization, Add
from tensorflow.keras.layers import Attention, LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pickle

from sklearn.ensemble import IsolationForest

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_csv_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(i) if i else np.nan for i in row])
    return np.array(data)

def impute_missing_values(data, method='mean', n_neighbors=5):
    if method == 'mean':
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    elif method == 'median':
        col_median = np.nanmedian(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_median, inds[1])
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data = imputer.fit_transform(data)
    return data


def plot_outliers(data, mask, method):
    """Helper function to plot original data and detected outliers."""
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(data)), data, color='blue', label='Original Data')
    plt.scatter(np.arange(len(data))[~mask], data[~mask], color='red', label='Outliers')
    plt.title(f'Outlier Detection using {method}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show(block=True)


def remove_outliers(data, method='zscore', z_thresh=3.0, is_plot=True):
    if method == 'zscore':
        z_scores = np.abs((data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0))
        mask = (z_scores < z_thresh).all(axis=1)
    elif method == 'isolation_forest':
        iso = IsolationForest(contamination=0.01)
        yhat = iso.fit_predict(data)
        mask = yhat != -1
    else:
        return data  # Return original data if method is not recognized
    filtered_data = data[mask]
    if is_plot:
        plot_outliers(data, mask, method)
    return filtered_data


def normalize_data(data, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    elif method == 'standard':
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    elif method == 'robust':
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
        scaler = None
    return data_scaled, scaler

def compute_difference(data,is_plot = True):
    # Compute the difference (delta) of the data
    diff_data = np.diff(data, axis=0, prepend=data[0:1])

    if is_plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # Plot original data
        axs[0].plot(data, label='Original Data', color='blue')
        axs[0].set_title('Original Data')
        axs[0].legend()
        # Plot differences
        axs[1].plot(diff_data, label='Differences', color='orange')
        axs[1].set_title('Differences')
        axs[1].legend()
        # Set common x-axis label
        axs[1].set_xlabel('Index')
        # Show the plot
        plt.tight_layout()
        plt.show(block=True)
 

    return diff_data

def de_difference(diff_data, initial_value):
    # Reconstruct the original data from differenced data
    original_data = np.cumsum(diff_data, axis=0) + initial_value
    return original_data

def extract_time_domain_features(window):
    # Basic statistics
    min_feat = np.min(window, axis=0)
    max_feat = np.max(window, axis=0)
    std_feat = np.std(window, axis=0)
    mean_feat = np.mean(window, axis=0)
    median_feat = np.median(window, axis=0)
    skewness_feat = skew(window, axis=0)
    kurtosis_feat = kurtosis(window, axis=0)
    iqr_feat = iqr(window, axis=0)
    # Advanced time-domain features
    energy_feat = np.sum(np.square(window), axis=0)  # Signal energy
    entropy_feat = entropy(np.abs(window), axis=0)  # Entropy of the signal
    # Quantile-based features (25th, 75th percentiles)
    quantile_25 = np.percentile(window, 25, axis=0)
    quantile_75 = np.percentile(window, 75, axis=0)
    range_feat = max_feat - min_feat  # Signal range
    # Combine all features
    features = np.concatenate((
        min_feat, max_feat, std_feat, mean_feat, median_feat, skewness_feat,
        kurtosis_feat, iqr_feat, energy_feat, entropy_feat, quantile_25, quantile_75, range_feat
    ))
    return features


def extract_frequency_domain_features(window, roll_off=0.85):
    # FFT transformation and magnitudes
    fft_values = fft(window, axis=0)
    fft_magnitude = np.abs(fft_values)
    # Use only half of the spectrum due to symmetry
    half_spectrum = fft_magnitude[:len(fft_magnitude) // 2]
    # Basic frequency-domain features
    mean_fft = np.mean(half_spectrum, axis=0)
    std_fft = np.std(half_spectrum, axis=0)
    # Advanced frequency-domain features
    dominant_freq = np.argmax(half_spectrum, axis=0)  # Index of the peak frequency component
    spectral_entropy = entropy(half_spectrum, axis=0)  # Entropy in the frequency domain
    # Spectral roll-off (frequency below which a specified percentage of spectral energy is contained)
    roll_off_freq = np.argmax(np.cumsum(half_spectrum, axis=0) >= roll_off * np.sum(half_spectrum, axis=0), axis=0)
    # Combine frequency features
    freq_features = np.concatenate((mean_fft, std_fft, dominant_freq, spectral_entropy, roll_off_freq))
    return freq_features


def extract_features(window, params):
    features_list = []
    if params['time_domain_features']:
        time_features = extract_time_domain_features(window)
        features_list.append(time_features)
    if params['frequency_domain_features']:
        freq_features = extract_frequency_domain_features(window)
        features_list.append(freq_features)
    if params['use_original']:
        original_features = window.flatten()
        features_list.append(original_features)
    features = np.concatenate(features_list)
    return features

def create_windows(data, params):
    X, y = [], []
    initial_values = []
    window_size = params['window_size']
    num_features = data.shape[1]
    for i in tqdm(range(len(data) - window_size), desc="Processing windows"):
        window = data[i:(i + window_size), :]
        if params.get('use_difference', False):
            # When using differences, target is the difference between next value and last value in the window
            target = data[i + window_size, :] - data[i + window_size -1, :]
            initial_value = data[i + window_size -1, :]
            initial_values.append(initial_value)
        else:
            target = data[i + window_size, :]
        features_list = []
        # For multivariate data, process each feature individually
        for feature_idx in range(num_features):
            window_feature = window[:, feature_idx].reshape(-1, 1)
            features = extract_features(window_feature, params)
            features_list.append(features)
        # Concatenate features from all variables
        combined_features = np.concatenate(features_list)
        X.append(combined_features)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    if params.get('use_difference', False):
        return X, y, np.array(initial_values)
    else:
        return X, y

def build_lstm_model(input_shape,target_shape, units=64, dropout_rate=0.2, l2_reg=0.001):
    model = Sequential()
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
    model.add(LSTM(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(target_shape[-1]))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_simple_attention_model(input_shape,target_shape, units=64, dropout_rate=0.2, l2_reg=0.001):
    inputs = Input(shape=(input_shape[0], 1))
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    attention = Attention()([lstm_out, lstm_out])
    attention_flat = Flatten()(attention)
    outputs = Dense(target_shape[-1])(attention_flat)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model




def build_hybrid_cnn_attention_model(input_shape, target_shape, num_heads=4, dff=128, cnn_filters=64, kernel_size=3,
                                     dropout_rate=0.1):
    inputs = Input(shape=(input_shape[0], 1))

    # Convolutional Block
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(cnn_output)
    cnn_output = Dropout(dropout_rate)(cnn_output)

    # Multi-Head Attention Block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=cnn_filters)(cnn_output, cnn_output)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(Add()([cnn_output, attn_output]))  # Skip connection

    # Feed Forward Network
    ffn_output = Dense(dff, activation='relu')(attn_output)
    ffn_output = Dense(cnn_filters)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(Add()([attn_output, ffn_output]))  # Skip connection

    # Flatten and Output
    flat = Flatten()(ffn_output)
    outputs = Dense(target_shape[-1])(flat)

    # Model Compilation
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, params, scaler=None, initial_values_train=None, initial_values_test=None):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # De-normalize predictions if scaler is provided
    if scaler is not None:
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        if initial_values_train is not None:
            initial_values_train = scaler.inverse_transform(initial_values_train)
        if initial_values_test is not None:
            initial_values_test = scaler.inverse_transform(initial_values_test)

    if params.get('use_difference', False):
        # De-difference the predictions
        y_train_de_diff = initial_values_train + y_train
        y_train_pred_de_diff = initial_values_train + y_train_pred
        y_test_de_diff = initial_values_test + y_test
        y_test_pred_de_diff = initial_values_test + y_test_pred
    else:
        y_train_de_diff = y_train
        y_train_pred_de_diff = y_train_pred
        y_test_de_diff = y_test
        y_test_pred_de_diff = y_test_pred

    # Now compute metrics using y_train_de_diff, y_train_pred_de_diff, etc.

    metrics = {
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'R2_Score': r2_score
    }

    results = {}
    for name, func in metrics.items():
        train_metric = func(y_train_de_diff, y_train_pred_de_diff)
        test_metric = func(y_test_de_diff, y_test_pred_de_diff)
        results[name] = {'train': train_metric, 'test': test_metric}

    print(results)

    # Save results
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_metrics_{date_time}.txt")
    with open(results_file, 'w') as f:
        for metric_name, values in results.items():
            f.write(f"{metric_name} - Train: {values['train']}, Test: {values['test']}\n")

    # Save the model and parameters
    model_file = os.path.join(models_dir, f"{model_name}_{date_time}.h5")
    model.save(model_file)
    params_file = os.path.join(models_dir, f"{model_name}_params_{date_time}.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)

    # Plot actual vs predicted
    plt.figure(figsize=(12,6))
    plt.plot(y_test_de_diff.flatten(), label='Actual')
    plt.plot(y_test_pred_de_diff.flatten(), label='Predicted')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{model_name}_prediction_plot_{date_time}.png"))
    plt.show(block=True)

    return results, model_file, params_file

def plot_training_history(history, model_name):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=True)


def future_prediction(model, data, original_data, params, scaler=None):
    predictions = []
    window_size = params['window_size']
    num_features = data.shape[1]
    input_seq = data[-window_size:, :]

    if params.get('use_difference', False):
        # Store the last actual value from original data for de-differencing
        last_actual_value = original_data[-1, :]

    for _ in range(params['horizon']):
        features_list = []
        for feature_idx in range(num_features):
            window_feature = input_seq[:, feature_idx].reshape(-1, 1)
            features = extract_features(window_feature, params)
            features_list.append(features)

        combined_features = np.concatenate(features_list)
        input_seq_reshaped = combined_features.reshape((1, -1))
        pred = model.predict(input_seq_reshaped)

        # if params.get('use_difference', False):

            # if scaler is not None:
            #     # De-normalize the predicted difference
            #     pred_denorm = scaler.inverse_transform(pred)
            #     # Add to the last actual value to get the real prediction
            #     actual_pred = last_actual_value + pred_denorm
            #     # Update last actual value for next iteration
            #     last_actual_value = actual_pred
            #     # Normalize back for the sliding window
            #     new_input = scaler.transform(actual_pred)
            # else:
            #     # If no scaler, just add the predicted difference
            #     actual_pred = last_actual_value + pred
            #     last_actual_value = actual_pred
            #     new_input = actual_pred
            #
            # predictions.append(actual_pred[0])
        # else:
        # For non-differenced data
        if scaler is not None:
            pred_denorm = scaler.inverse_transform(pred)
            predictions.append(pred_denorm[0])
            new_input = pred  # Keep normalized for sliding window
        else:
            predictions.append(pred[0])
            new_input = pred

        input_seq = np.vstack((input_seq[1:], new_input))


    predictions = np.array(predictions)

    # new ------------
    if params.get('use_difference', False):
        predictions = de_difference(predictions, last_actual_value)
     # if scaler is not None:
    #     # De-normalize the predicted difference
    #     pred_denorm = scaler.inverse_transform(pred)
    #     # Add to the last actual value to get the real prediction
    #     actual_pred = last_actual_value + pred_denorm
    #     # Update last actual value for next iteration
    #     last_actual_value = actual_pred
    #     # Normalize back for the sliding window
    #     new_input = scaler.transform(actual_pred)
    # else:
    #     # If no scaler, just add the predicted difference
    #     actual_pred = last_actual_value + pred
    #     last_actual_value = actual_pred
    #     new_input = actual_pred
    #
    # predictions.append(actual_pred[0])

    # ----------------


    # ----- plot the future forecast
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(original_data)), original_data, label='Historical Data')
    plt.plot(range(len(original_data), len(original_data) + len(predictions)), predictions, label='Future Predictions')
    plt.title(f' - Future Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join('results', f"_future_prediction.png"))
    plt.show(block=True)
    return predictions

def future_prediction_with_new_data(model_path, params_path, data_path):
    # Load model and parameters
    model = load_model(model_path)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # Load new data
    new_data = load_csv_data(data_path)

    # Preprocess new data
    if params['imputation_method']:
        new_data = impute_missing_values(new_data, method=params['imputation_method'],
                                         n_neighbors=params['imputation_neighbors'])
    # Apply difference if required
    if params.get('use_difference', False):
        new_data = compute_difference(new_data)

    if params['outlier_removal']:
        new_data = remove_outliers(new_data, method=params['outlier_method'])


    if params['normalization_method']:
        scaler = params['scaler']
        if scaler:
            new_data = scaler.transform(new_data)
        else:
            new_data, _ = normalize_data(new_data, method=params['normalization_method'])
    else:
        scaler = None

    # Ensure data is in correct shape
    if len(new_data.shape) == 1 or new_data.shape[1] == 1:
        new_data = new_data.reshape(-1, 1)

    # Perform future prediction
    predictions = future_prediction(
        model,
        new_data,
        new_data,
        params,
        scaler
    )
    # De-normalize predictions if scaler is provided
    if scaler is not None:
        pass  # Predictions are already de-normalized in future_prediction
    print("Future Predictions (new data):", predictions)
    return predictions

def plot_train_test_split(data, train_indices, test_indices):
    plt.figure(figsize=(12,6))
    plt.plot(train_indices, data[train_indices].flatten(), label='Train Data', color='blue')
    plt.plot(test_indices, data[test_indices].flatten(), label='Test Data', color='orange')
    plt.title('Train-Test Split')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show(block=True)

def main():
    # Parameters (can be set as needed)
    params = {
        'file_path': 'data_xauusd_2.csv',  # Path to your main training data CSV file
        'window_size': 600,
        'normalization_method': 'None',  # Options: 'minmax', 'standard', 'robust', None
        'imputation_method': 'knn',  # Options: 'mean', 'median', 'knn', None
        'imputation_neighbors': 5,  # Number of neighbors for KNN imputation
        'outlier_removal': False,
        'outlier_method': 'zscore',  # Options: 'zscore', 'isolation_forest', None
        'feature_extraction': True,
        'time_domain_features': True,
        'frequency_domain_features': True,
        'use_original': True,  # Whether to use the original data as features
        'use_difference': True,  # Whether to use the difference of data instead of original data.first-order differencing
        'epochs': 10,
        'batch_size': 32,
        'patience': 30,
        'horizon': 45,
        'retrain': True,  # Set to True to retrain saved model
        'model_path': None,  # Path to saved model (if retrain is False)
        'params_path': None,  # Path to saved parameters (if retrain is False)
        'split_method': 'time_series',  # Options: 'random', 'time_series'
        'n_splits': 5  # For TimeSeriesSplit
    }

    # Load data
    data = load_csv_data(params['file_path'])
    original_data = data.copy()  # Save original data for de-differencing

    # Preprocessing
    if params['imputation_method']:
        data = impute_missing_values(data, method=params['imputation_method'],
                                     n_neighbors=params['imputation_neighbors'])
    # Apply difference if required
    if params.get('use_difference', False):
        data = compute_difference(data)

    if params['outlier_removal']:
        data = remove_outliers(data, method=params['outlier_method'])


    if params['normalization_method']:
        data, scaler = normalize_data(data, method=params['normalization_method'])
        params['scaler'] = scaler
    else:
        scaler = None
        params['scaler'] = None

    # Detect univariate or multivariate
    if len(data.shape) == 1 or data.shape[1] == 1:
        data = data.reshape(-1, 1)

    # Create windows
    if params.get('use_difference', False):
        X, y, initial_values = create_windows(data, params)
    else:
        X, y = create_windows(data, params)

    # Determine input shape
    input_shape = (X.shape[1],)
    target_shape = (y.shape[1],)

    # Split data
    if params['split_method'] == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
        if params.get('use_difference', False):
            initial_values_train, initial_values_test = train_test_split(
                initial_values, test_size=0.2, shuffle=True)
        # Plot train and test data
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
    elif params['split_method'] == 'time_series':
        tscv = TimeSeriesSplit(n_splits=params['n_splits'])
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if params.get('use_difference', False):
            initial_values_train, initial_values_test = initial_values[train_index], initial_values[test_index]
        # Plot train and test data
        train_indices = train_index
        test_indices = test_index
    else:
        raise ValueError("Invalid split method specified.")

    plot_train_test_split(y, train_indices, test_indices)
    # 'Simple_LSTM': build_lstm_model,
    # 'Simple_Attention': build_simple_attention_model,
    # 'Simple_Sophisticated_Attention': build_sophisticated_attention_model,

    models = {
        'Simple_Sophisticated_Attention': build_hybrid_cnn_attention_model,
    }

    trained_models = {}
    for model_name, model_builder in models.items():
        print(f"Training {model_name}...")
        model = model_builder(input_shape,target_shape)
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        # Train model
        history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                            validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
        # Plot training history
        plot_training_history(history, model_name)
        # Evaluate model and save
        if params.get('use_difference', False):
            results, model_file, params_file = evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name, params, scaler, initial_values_train, initial_values_test)
        else:
            results, model_file, params_file = evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name, params, scaler)
        print(f"{model_name} Evaluation Results:", results)
        trained_models[model_name] = {
            'model': model,
            'model_file': model_file,
            'params_file': params_file
        }

        # Future prediction using the same data
        predictions = future_prediction(model, data, original_data, params, scaler)



    return
    # Future prediction with new data
    new_data_path = 'data3.csv'  # Replace with your new data file path
    for model_name, model_info in trained_models.items():
        print(f"Future prediction with new data using {model_name}...")
        predictions = future_prediction_with_new_data(
            model_path=model_info['model_file'],
            params_path=model_info['params_file'],
            data_path=new_data_path
        )
        print(f"{model_name} Predictions on New Data:", predictions)

if __name__ == "__main__":
    main()
