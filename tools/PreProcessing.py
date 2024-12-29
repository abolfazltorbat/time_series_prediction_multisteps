import pywt
import os
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import skew, kurtosis, iqr, entropy
from scipy.fft import rfft
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split, TimeSeriesSplit

def check_for_saved_params(params, logger):
    params_path = params.get('params_path', None)
    if params_path and os.path.exists(params_path):
        logger.info(f"Loading parameters from {params_path}")
        with open(params_path, 'rb') as f:
            loaded_params = pickle.load(f)

        # Overwrite some critical params if they are defined in the current run
        if params.get('model_path', None) is not None:
            loaded_params['model_path'] = params['model_path']
        if params.get('file_path', None) is not None:
            loaded_params['file_path'] = params['file_path']
        if params.get('epochs', None) is not None:
            loaded_params['epochs'] = params['epochs']
        if params.get('batch_size', None) is not None:
            loaded_params['batch_size'] = params['batch_size']
        if params.get('patience', None) is not None:
            loaded_params['patience'] = params['patience']
        if params.get('horizon', None) is not None:
            loaded_params['horizon'] = params['horizon']
        if params.get('n_splits', None) is not None:
            loaded_params['n_splits'] = params['n_splits']
        if params.get('alpha_sections', None) is not None:
            loaded_params['alpha_sections'] = params['alpha_sections']
        if params.get('is_plot', None) is not None:
            loaded_params['is_plot'] = params['is_plot']
        if params.get('optimizer', None) is not None:
            loaded_params['optimizer'] = params['optimizer']
        if params.get('results_save_dir', None) is not None:
            loaded_params['results_save_dir'] = params['results_save_dir']

        params = loaded_params
    else:
        if params_path:
            logger.warning(f"params_path {params_path} not found. Using default parameters.")
        else:
            logger.info("No external params_path provided. Using default parameters.")

    return params

def preprocess_data(params, logger):
    """Preprocess the data according to specified parameters."""
    data = load_csv_data(params['file_path'], logger)
    original_data = data.copy()
    if params['imputation_method']:
        data = impute_missing_values(
            data,
            method=params['imputation_method'],
            n_neighbors=params['imputation_neighbors'],
            logger=logger
        )
    if params.get('use_difference', False):
        data = compute_difference(data, params=params)
    if params['outlier_removal']:
        data = remove_outliers(data, method=params['outlier_method'], params=params)
    if params['normalization_method']:
        data, scaler = normalize_data(data, method=params['normalization_method'], logger=logger)
        params['scaler'] = scaler
    else:
        scaler = None
        params['scaler'] = None
    if len(data.shape) == 1 or data.shape[1] == 1:
        data = data.reshape(-1, 1)
    if params.get('use_difference', False):
        X, y, initial_values = create_windows(data, params)
        return X, y, initial_values, data, original_data, scaler
    else:
        X, y = create_windows(data, params)
        return X, y, None, data, original_data, scaler

def load_csv_data(file_path, logger):
    """Load data from a CSV file and handle missing values."""
    logger.info(f"Loading CSV data from {file_path}")
    data = pd.read_csv(file_path).values
    zero_nan_mask = np.isnan(data) | (data == 0)
    if np.any(zero_nan_mask):
        logger.info("Data contains zeros or NaNs. Handling them appropriately.")
        data = impute_missing_values(data, method='mean', logger=logger)
    return data

def remove_outliers(data, method='zscore', z_thresh=3.0, is_plot=False, params=None):
    """Remove outliers from the data using the specified method."""
    logger = logging.getLogger('TimeSeriesModel')
    logger.info(f"Removing outliers using method: {method}")
    if method == 'zscore':
        z_scores = np.abs((data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0))
        mask = (z_scores < z_thresh).all(axis=1)
    elif method == 'isolation_forest':
        iso = IsolationForest(contamination=0.01)
        yhat = iso.fit_predict(data)
        mask = yhat != -1
    else:
        return data
    filtered_data = data[mask]
    if False and is_plot:
        plot_outliers(data, mask, method, params)
    return filtered_data
def plot_outliers(data, mask, method, params):
    """Plot outliers detected in the data."""
    logger = logging.getLogger('TimeSeriesModel')
    logger.info(f"Plotting outliers detected by method: {method}")
    plt.figure(figsize=(12, 6), dpi=params['dpi'])
    plt.plot(data, label='Original Data')
    plt.plot(np.where(mask, data, np.nan), 'ro', label='Outliers')
    plt.title(f'Outlier Detection using {method}')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(params['results_save_dir'], f"outliers_{method}.svg")
    plt.savefig(save_path, format='svg')
    if params['is_plot']:
        plt.show()
    plt.close()

def normalize_data(data, method='minmax', logger=None):
    """Normalize the data using the specified method."""
    logger.info(f"Normalizing data using method: {method}")
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

def impute_missing_values(data, method='mean', n_neighbors=5, logger=None):
    """Impute missing values in the data using the specified method."""
    logger.info(f"Imputing missing values using method: {method}")
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

def compute_difference(data, is_plot=False, params=None):
    """
    Compute and visualize the first-order difference of the data with academic formatting.

    Args:
        data: Input time series data
        is_plot: Boolean flag for plotting
        params: Dictionary containing plotting parameters

    Returns:
        diff_data: First-order differenced data
    """
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Computing first-order difference of the data")
    diff_data = np.diff(data, axis=0, prepend=data[0:1])

    if is_plot:
        # Set the style for academic plotting
        plt.style.use('grayscale')

        # Create figure and subplots with higher DPI
        fig, axs = plt.subplots(2, 1, figsize=(15, 8),
                                sharex=True, dpi=100,
                                gridspec_kw={'hspace': 0.3})

        # Plot original data
        axs[0].plot(data,
                    label='Original Data',
                    color='#2C3E50',  # Dark blue
                    linewidth=1.0)
        axs[0].set_title('Original Time Series',
                         fontsize=10,
                         fontweight='bold',
                         pad=10)

        # Plot differenced data
        axs[1].plot(diff_data,
                    label='First-Order Difference',
                    color='#E74C3C',  # Professional red
                    linewidth=1.0)
        axs[1].set_title('Differenced Time Series',
                         fontsize=10,
                         fontweight='bold',
                         pad=10)

        # Customize both subplots
        for ax in axs:
            # Enhance grid
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
            ax.minorticks_on()

            # Customize spines
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            # Format axis labels
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Add legend with enhanced styling
            ax.legend(loc='upper right',
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      fontsize=10)

            # Scientific notation if needed
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

        # Add main title and axis labels
        # fig.suptitle('Comparison of Original and Differenced Time Series',
        #              fontsize=10,
        #              fontweight='bold',
        #              y=0.95)

        fig.supxlabel('Time Steps',
                      fontsize=12,
                      fontweight='bold',
                      y=0.02)

        fig.supylabel('Consumption(MW)',
                      fontsize=12,
                      fontweight='bold',
                      x=0.02)

        # Adjust layout
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])

        # Save the figure with high quality
        save_path = os.path.join(params['results_save_dir'], "difference_plot.png")
        plt.savefig(save_path,
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0.1,
                    metadata={'Creator': 'Difference Plot'})

        if params['is_plot']:
            plt.show()
        plt.close()

    return diff_data

def de_difference(diff_data, initial_value):
    """Reconstruct original data from differenced data."""
    original_data = np.cumsum(diff_data, axis=0) + initial_value
    return original_data

def zero_crossing_rate(window):
    signs = np.sign(window)
    zero_crossings = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
    return zero_crossings / window.shape[0]

def mean_absolute_deviation(window):
    return np.mean(np.abs(window - np.mean(window, axis=0)), axis=0)

def compute_hjorth_parameters(window):
    first_derivative = np.diff(window, axis=0)
    second_derivative = np.diff(first_derivative, axis=0)
    variance0 = np.var(window, axis=0)
    variance1 = np.var(first_derivative, axis=0)
    variance2 = np.var(second_derivative, axis=0)
    activity = variance0
    mobility = np.sqrt(variance1 / (variance0 + 1e-8))
    complexity = np.sqrt(variance2 / (variance1 + 1e-8)) / (mobility + 1e-8)
    return activity, mobility, complexity

def select_recent_data(data):
    data_length = len(data)
    if 1 <= data_length <= 50:
        fraction = 0.1
    elif 51 <= data_length <= 500:
        fraction = 0.05
    elif 501 <= data_length <= 5000:
        fraction = 0.01
    else:
        fraction = 0.005
    num_samples = max(1, int(data_length * fraction))
    selected_data = data[-num_samples:].reshape(-1)
    return selected_data
def extract_time_domain_features(window):
    stats = {
        'min': np.min(window, axis=0),
        'max': np.max(window, axis=0),
        'std': np.std(window, axis=0),
        'mean': np.mean(window, axis=0),
        'median': np.median(window, axis=0),
        'iqr': iqr(window, axis=0),
        'variance': np.var(window, axis=0),
        'energy': np.sum(window ** 2, axis=0),
        'rms': np.sqrt(np.mean(window ** 2, axis=0)),
    }
    skewness_feat = skew(window, axis=0)
    kurtosis_feat = kurtosis(window, axis=0)
    entropy_feat = entropy(np.abs(window) + 1e-8, axis=0)
    quantiles = np.percentile(window, [25, 75], axis=0)
    range_feat = stats['max'] - stats['min']
    mad_feat = mean_absolute_deviation(window)
    zcr_feat = zero_crossing_rate(window)
    activity_feat, mobility_feat, complexity_feat = compute_hjorth_parameters(window)
    last_data = select_recent_data(window)

    # Ensure all features have the same dimensions (1D) before concatenation
    feature_list = [
        last_data.reshape(-1),  # Convert to 1D
        stats['min'].reshape(-1),
        stats['max'].reshape(-1),
        stats['std'].reshape(-1),
        stats['mean'].reshape(-1),
        stats['median'].reshape(-1),
        skewness_feat.reshape(-1),
        kurtosis_feat.reshape(-1),
        stats['iqr'].reshape(-1),
        stats['energy'].reshape(-1),
        entropy_feat.reshape(-1),
        quantiles[0].reshape(-1),  # 25th percentile
        quantiles[1].reshape(-1),  # 75th percentile
        range_feat.reshape(-1),
        stats['variance'].reshape(-1),
        stats['rms'].reshape(-1),
        mad_feat.reshape(-1),
        zcr_feat.reshape(-1),  # Corrected line
        activity_feat.reshape(-1),
        mobility_feat.reshape(-1),
        complexity_feat.reshape(-1)
    ]

    # Concatenate all features into a single array
    features = np.concatenate(feature_list)
    return features

def spectral_flatness(half_spectrum):
    geometric_mean = np.exp(np.mean(np.log(half_spectrum + 1e-12), axis=0))
    arithmetic_mean = np.mean(half_spectrum, axis=0)
    return geometric_mean / (arithmetic_mean + 1e-12)

def spectral_flux(half_spectrum):
    diff = np.diff(half_spectrum, axis=0)
    flux = np.sqrt(np.sum(diff ** 2, axis=0)) / half_spectrum.shape[0]
    return flux

def extract_frequency_domain_features(window, roll_off=0.85):
    fft_values = rfft(window, axis=0)
    half_spectrum = np.abs(fft_values)
    sum_spectrum = np.sum(half_spectrum, axis=0) + 1e-8
    cumulative_sum = np.cumsum(half_spectrum, axis=0)

    spectral_centroid_full = np.sum(np.arange(half_spectrum.shape[0])[:, None] * half_spectrum, axis=0) / sum_spectrum
    spectral_bandwidth_full = np.sqrt(
        np.sum(((np.arange(half_spectrum.shape[0])[:, None] - spectral_centroid_full) ** 2) * half_spectrum, axis=0) / sum_spectrum
    )

    roll_off_freq = np.argmax(cumulative_sum >= roll_off * sum_spectrum, axis=0)
    spectral_entropy = entropy(half_spectrum + 1e-8, axis=0)
    flatness_feat = spectral_flatness(half_spectrum)
    flux_feat = spectral_flux(half_spectrum)

    spectral_centroid = np.array([
        np.mean(spectral_centroid_full),
        np.median(spectral_centroid_full),
        np.max(spectral_centroid_full),
        np.min(spectral_centroid_full),
        np.std(spectral_centroid_full)
    ])

    spectral_bandwidth = np.array([
        np.mean(spectral_bandwidth_full),
        np.median(spectral_bandwidth_full),
        np.max(spectral_bandwidth_full),
        np.min(spectral_bandwidth_full),
        np.std(spectral_bandwidth_full)
    ])

    freq_features = np.concatenate((
        np.mean(half_spectrum, axis=0),
        np.std(half_spectrum, axis=0),
        np.argmax(half_spectrum, axis=0),
        spectral_entropy,
        roll_off_freq,
        spectral_centroid,
        spectral_bandwidth,
        flatness_feat,
        flux_feat
    ))
    return freq_features

def extract_wavelet_features(window):
    coeffs = pywt.wavedec(window.flatten(), 'db1', level=3)
    coeffs_array = np.concatenate(coeffs)
    wavelet_features = np.array([
        np.mean(coeffs_array),
        np.std(coeffs_array),
        np.max(coeffs_array),
        np.min(coeffs_array)
    ])
    return wavelet_features

def extract_features(window, params):
    features_list = []
    window = window.reshape(-1, 1)
    if params['time_domain_features']:
        time_features = extract_time_domain_features(window)
        features_list.append(time_features)
    if params['frequency_domain_features']:
        freq_features = extract_frequency_domain_features(window)
        features_list.append(freq_features)
    if params['wavelet_features']:
        wavelet_features = extract_wavelet_features(window)
        features_list.append(wavelet_features)
    if params['use_original']:
        features_list.append(window.flatten())
    features = np.concatenate(features_list)
    return features

def process_window(i, data, params, window_size, num_features):
    window = data[i:(i + window_size), :]
    if params.get('use_difference', False):
        target = data[i + window_size, :] - data[i + window_size - 1, :]
        initial_value = data[i + window_size - 1, :]
    else:
        target = data[i + window_size, :]
        initial_value = None

    features_list = []
    for feature_idx in range(num_features):
        window_feature = window[:, feature_idx]
        features = extract_features(window_feature.reshape(-1, 1), params)
        features_list.append(features)
    combined_features = np.concatenate(features_list)
    return combined_features, target, initial_value

def create_windows(data, params):
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Creating windows for the model")
    window_size = params['window_size']
    num_features = data.shape[1]
    total_windows = len(data) - window_size


    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, data, params, window_size, num_features)
        for i in tqdm(range(total_windows), desc="Parallel Processing the windows")
    )

    X, y, initial_values = zip(*results)
    X = np.array(X)
    y = np.array(y)
    if params.get('use_difference', False):
        initial_values = np.array(initial_values)
        return X, y, initial_values
    else:
        return X, y

def split_data(X, y, initial_values, params, logger):
    """Split data into training and testing sets."""
    logger.info(f"Splitting data using method: {params['split_method']}")
    if params['split_method'] == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
        if params.get('use_difference', False):
            initial_values_train, initial_values_test = train_test_split(
                initial_values, test_size=0.2, shuffle=True)
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
        train_indices = train_index
        test_indices = test_index
    elif params['split_method'] == 'sectional':
        alpha = params['alpha_sections']
        section_size = len(X) // alpha
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        initial_values_train_list = []
        initial_values_test_list = []
        train_indices = []
        test_indices = []
        for i in range(alpha):
            start_idx = i * section_size
            end_idx = (i + 1) * section_size if i < alpha - 1 else len(X)
            X_section = X[start_idx:end_idx]
            y_section = y[start_idx:end_idx]
            if params.get('use_difference', False):
                initial_values_section = initial_values[start_idx:end_idx]
            split_point = int(0.8 * len(X_section))
            X_train_section = X_section[:split_point]
            X_test_section = X_section[split_point:]
            y_train_section = y_section[:split_point]
            y_test_section = y_section[split_point:]
            X_train_list.append(X_train_section)
            X_test_list.append(X_test_section)
            y_train_list.append(y_train_section)
            y_test_list.append(y_test_section)
            if params.get('use_difference', False):
                initial_values_train_list.append(initial_values_section[:split_point])
                initial_values_test_list.append(initial_values_section[split_point:])
            train_indices.extend(range(start_idx, start_idx + split_point))
            test_indices.extend(range(start_idx + split_point, end_idx))
        X_train = np.concatenate(X_train_list)
        X_test = np.concatenate(X_test_list)
        y_train = np.concatenate(y_train_list)
        y_test = np.concatenate(y_test_list)
        if params.get('use_difference', False):
            initial_values_train = np.concatenate(initial_values_train_list)
            initial_values_test = np.concatenate(initial_values_test_list)
    else:
        raise ValueError("Invalid split method specified.")

    # Plot train-test split if needed
    if params.get('is_plot', False):
        plot_train_test_split(y, train_indices, test_indices, params)

    if params.get('use_difference', False):
        return X_train, X_test, y_train, y_test, train_indices, test_indices, initial_values_train, initial_values_test
    else:
        return X_train, X_test, y_train, y_test, train_indices, test_indices, None, None
def plot_train_test_split(data, train_indices, test_indices, params):
    """
    Plot the train-test split of the data with academic paper formatting.

    Args:
        data: Input time series data
        train_indices: Indices for training data
        test_indices: Indices for test data
        params: Dictionary containing plotting parameters
    """
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Plotting train-test split")

    # Set the style for academic plotting
    plt.style.use('grayscale')

    # Create figure and axis objects with higher DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot complete dataset
    total_indices = np.arange(len(data))
    ax.plot(total_indices,
            data.flatten(),
            label='Complete Dataset',
            color='#7F8C8D',  # Professional gray
            linestyle='-',
            linewidth=1.0,
            alpha=0.3)

    # Plot train and test segments
    color_scheme = {
        'Train Data': '#2C3E50',  # Dark blue
        'Test Data': '#E74C3C'  # Professional red
    }

    for indices, label, color in [
        (train_indices, 'Train Data', color_scheme['Train Data']),
        (test_indices, 'Test Data', color_scheme['Test Data']),
    ]:
        segments = get_continuous_segments(indices)
        for start, end in segments:
            idx_range = np.arange(start, end + 1)
            data_segment = data[idx_range].flatten()
            ax.plot(
                idx_range,
                data_segment,
                label=label if (start == segments[0][0]) else "",
                color=color,
                linestyle='-',
                linewidth=1.5
            )

    # Customize title and labels
    ax.set_title('Temporal Data Split: Training and Testing Sets',
                 fontsize=14,
                 fontweight='bold',
                 pad=20)

    ax.set_xlabel('Time Steps',
                  fontsize=12,
                  fontweight='bold')

    ax.set_ylabel('Value',
                  fontsize=12,
                  fontweight='bold')

    # Format axis ticks
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=10)

    # Customize grid
    ax.grid(True,
            linestyle='--',
            alpha=0.7)
    ax.grid(True,
            which='minor',
            linestyle=':',
            alpha=0.3)
    ax.minorticks_on()

    # Enhanced legend
    ax.legend(loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=10,
              ncol=1)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Scientific notation for y-axis if needed
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    # Add text box with split information
    split_ratio = len(train_indices) / (len(train_indices) + len(test_indices))
    text_str = f'Split Ratio (Train/Total): {split_ratio:.2%}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props)

    # Tight layout
    plt.tight_layout()

    # Save the figure with high quality
    save_path = os.path.join(params['results_save_dir'], "train_test_split.png")
    plt.savefig(save_path,
                format='png',
                bbox_inches='tight',
                pad_inches=0.1,
                metadata={'Creator': 'Train-Test Split Plot'})

    if params['is_plot']:
        plt.show()
    plt.close()
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

