import logging
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import mpld3
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import os
import numpy as np
from tools.BuildModel import (simple_cnn_lstm_model, deep_cnn_lstm_model, transformer_positional,
                               tcn_model, tcn_attention)
import seaborn as sns
import glob
import re
import pickle
from keras.saving import register_keras_serializable
##########################################
# Visualization Classes and Functions
##########################################

class ModelVisualizer:
    """
    A class to handle various model visualization tasks including architecture
    and training metrics.
    """

    def __init__(self, save_dir='./model'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_model_architecture(self, model, filename='model_architecture.png',
                                show_shapes=True, show_layer_names=True,
                                show_layer_activations=True, dpi=300):
        """
        Plot the model architecture using tensorflow's plot_model utility.
        """
        filepath = os.path.join(self.save_dir, filename)
        plot_model(model, to_file=filepath, show_shapes=show_shapes,
                   show_layer_names=show_layer_names,
                   show_layer_activations=show_layer_activations,
                   dpi=dpi)
        print(f"Model architecture saved to: {filepath}")

    def plot_training_history(self, history, metrics=None, figsize=(12, 8)):
        """
        Plot training history metrics.
        """
        if isinstance(history, tf.keras.callbacks.History):
            history = history.history

        if metrics is None:
            metrics = [m for m in history.keys() if not m.startswith('val_')]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            ax.plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')

            ax.set_title(f'{metric.upper()} Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        history_filepath = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(history_filepath)
        plt.close()
        print(f"Training history plot saved to: {history_filepath}")

    def plot_layer_output_distributions(self, model, sample_input, figsize=(15, 10)):
        """
        Plot the distribution of layer outputs for a sample input.
        """
        layer_outputs = []
        layer_names = []

        # Create intermediate models for each layer to extract outputs
        for layer in model.layers:
            try:
                intermediate_model = tf.keras.Model(inputs=model.input,
                                                    outputs=layer.output)
                output = intermediate_model.predict(sample_input)
                layer_outputs.append(output.flatten())
                layer_names.append(layer.name)
            except Exception as e:
                print(f"Skipping layer {layer.name}: {e}")
                continue

        # Plot distributions using seaborn
        n_layers = len(layer_outputs)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
            sns.histplot(output, bins=50, ax=axes[i])
            axes[i].set_title(f'Layer: {name}')
            axes[i].set_xlabel('Activation Value')

        # Remove empty subplots if any
        for i in range(n_layers, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        distribution_filepath = os.path.join(self.save_dir, 'layer_distributions.png')
        plt.savefig(distribution_filepath)
        plt.close()
        print(f"Layer output distributions saved to: {distribution_filepath}")


def visualize_model_training(model, history, sample_input=None,
                             save_dir='./visualizations'):
    """
    Wrapper function to generate all visualizations for a model.
    """
    visualizer = ModelVisualizer(save_dir=save_dir)

    # Plot model architecture
    visualizer.plot_model_architecture(model)

    # Plot training history
    visualizer.plot_training_history(history)

    # Plot layer distributions if sample input is provided
    if sample_input is not None:
        visualizer.plot_layer_output_distributions(model, sample_input)

    return visualizer


##########################################
# Custom Callbacks, Metrics and Losses
##########################################
class CustomModelCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback to save the best model based on validation loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@tf.keras.utils.register_keras_serializable()
class MSEComponent(tf.keras.metrics.Metric):
    """Custom metric to compute Mean Squared Error over batches."""

    def __init__(self, name='mse_component', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name='mse_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, y_pred.dtype)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        self.count.assign_add(1.0)

    def result(self):
        return self.mse_sum / self.count

    def reset_state(self):
        self.mse_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class CustomLoss(tf.keras.losses.Loss):
    """
    Custom loss function for multi-step financial time series prediction (e.g., XAU/USD).
    Components:
    1. MSE for overall accuracy.
    2. Directional accuracy with smoothing for trend prediction.
    3. Volatility matching to capture price fluctuations.
    """

    def __init__(self, direction_weight=1.0, volatility_weight=0.5, reduction='sum_over_batch_size',
                 name='financial_ts_loss'):
        super().__init__(reduction=reduction, name=name)
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        # Ensure shapes: (batch_size, 90, 1)
        if len(y_pred.shape) == 2:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)

        # 1. Mean Squared Error (MSE)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Directional accuracy with smoothing
        kernel = tf.constant([0.25, 0.5, 0.25], dtype=y_true.dtype)
        kernel = tf.reshape(kernel, [3, 1, 1])
        smoothed_y_true = tf.nn.conv1d(y_true, filters=kernel, stride=1, padding='SAME')
        smoothed_y_pred = tf.nn.conv1d(y_pred, filters=kernel, stride=1, padding='SAME')
        true_diff = smoothed_y_true[:, 1:, :] - smoothed_y_true[:, :-1, :]
        pred_diff = smoothed_y_pred[:, 1:, :] - smoothed_y_pred[:, :-1, :]
        directional_agreement = tf.nn.sigmoid(5.0 * true_diff * pred_diff)
        direction_loss = tf.reduce_mean(1.0 - directional_agreement)

        # 3. Volatility matching
        true_volatility = tf.math.reduce_std(true_diff, axis=1)
        pred_volatility = tf.math.reduce_std(pred_diff, axis=1)
        volatility_loss = tf.reduce_mean(tf.square(true_volatility - pred_volatility))

        total_loss = mse_loss + self.direction_weight * direction_loss + self.volatility_weight * volatility_loss
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "direction_weight": self.direction_weight,
            "volatility_weight": self.volatility_weight
        })
        return config

# Custom directional accuracy metric
@tf.keras.utils.register_keras_serializable()
class DirectionalAccuracy(tf.keras.metrics.Metric):
        def __init__(self, name='directional_accuracy', **kwargs):
            super().__init__(name=name, **kwargs)
            self.correct_directions = self.add_weight(name='correct', initializer='zeros')
            self.total = self.add_weight(name='total', initializer='zeros')

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, y_pred.dtype)
            true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]
            pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            agreement = tf.cast(tf.sign(true_diff) == tf.sign(pred_diff), tf.float32)
            self.correct_directions.assign_add(tf.reduce_mean(agreement))
            self.total.assign_add(1.0)

        def result(self):
            return self.correct_directions / self.total

        def reset_state(self):
            self.correct_directions.assign(0.0)
            self.total.assign(0.0)

        def get_config(self):
            return super().get_config()

##########################################
# Training Function
##########################################
def train_model_full_data(model, X_train, y_train, X_val, y_val, params, logger, initial_epoch=0,
                          is_train_new_model=True):
    """
    Train a transformer model for XAU/USD time series prediction with fast convergence and state-of-the-art techniques.
    """
    logger.info("Training transformer model for XAU/USD prediction")

    # Reshape targets to match model output: (batch_size, 90, 1)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))

    results_dir = params['results_save_dir']
    model_name = params.get('model_name', 'transformer_xauusd')

    # Paths for saving models
    best_model_path = os.path.join(results_dir, f"{model_name}_best.keras")
    final_model_path = os.path.join(results_dir, f"{model_name}_final.keras")
    epoch_model_path = os.path.join(results_dir, f"{model_name}_epoch_{{epoch:02d}}.keras")

    # Custom loss function
    custom_loss = CustomLoss(
        direction_weight=params.get('direction_weight', 1.0),
        volatility_weight=params.get('volatility_weight', 0.5)
    )


    # Cosine annealing learning rate schedule
    def cosine_annealing_schedule(epoch, lr, epochs=params['epochs'], lr_min=1e-6, lr_max=1e-3):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        return lr_min + (lr_max - lr_min) * cosine_decay

    # Callbacks
    checkpoint_callback = CustomModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    epoch_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=epoch_model_path,
        save_weights_only=False,
        verbose=1
    )

    callbacks = [
        checkpoint_callback,
        epoch_checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(cosine_annealing_schedule)
    ]

    # Enable mixed precision training for faster convergence
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if is_train_new_model:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=params.get('learning_rate', 1e-3),
            weight_decay=1e-4,
            clipnorm=params.get('gradient_clip_norm', 1.0)
        )

        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            MSEComponent(),
            DirectionalAccuracy()
        ]

        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=metrics
        )

    val_batch_size = params.get('validation_batch_size', params['batch_size'])

    try:
        history = model.fit(
            X_train,
            y_train,
            initial_epoch=initial_epoch,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            validation_batch_size=val_batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Preserve temporal order for time series
        )

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    # Save training history
    history_path = os.path.join(results_dir, f"{model_name}_history.npy")
    np.save(history_path, history.history)
    logger.info(f"Saved training history to {history_path}")

    # Save final model
    model.save(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")

    # Visualize training results
    sample_input = X_val[0:1] if X_val is not None else None
    visualize_model_training(
        model,
        history,
        sample_input=sample_input,
        save_dir=os.path.join(results_dir, 'visualizations')
    )

    # Reset mixed precision policy (optional)
    tf.keras.mixed_precision.set_global_policy('float32')

    return model, history

def find_model(folder):
    """
    Searches for a Keras model file in the given folder without knowing the model name.

    Priority:
      1. If any file matching "*_best_model.keras" exists, choose one (here we pick the most recently modified).
      2. If not, search for files matching "*_epoch_*.keras", extract epoch numbers, and choose the file with the highest epoch.

    Parameters:
      folder (str): The folder to search in.

    Returns:
      filename (str): The filename of the model that was loaded.
    """
    # 1. Look for best model files
    best_model_pattern = os.path.join(folder, "*_best_model.keras")
    best_files = glob.glob(best_model_pattern)

    if best_files:
        # If more than one best model file exists, we choose the one with the most recent modification time.
        best_files.sort(key=os.path.getmtime, reverse=True)
        chosen_file = best_files[0]
        print(f"Found best model file: {chosen_file}")
        try:
            return chosen_file
        except Exception as e:
            print(f"Error loading best model file {chosen_file}: {e}")
            return None

    # 2. If no best model file is found, look for epoch files.
    epoch_pattern = os.path.join(folder, "*_epoch_*.keras")
    epoch_files = glob.glob(epoch_pattern)

    if not epoch_files:
        print("No model files found matching the expected patterns in the folder.")
        return None

    # Prepare a regex pattern to extract the epoch number.
    # This expects filenames like "some_model_epoch_XX.keras" (e.g., transformer_gaussian_epoch_82.keras)
    epoch_regex = re.compile(r"(.+)_epoch_(\d{1,3})\.keras")
    max_epoch = -1
    chosen_file = None
    for file in epoch_files:
        base = os.path.basename(file)
        match = epoch_regex.match(base)
        if match:
            epoch_number = int(match.group(2))
            if epoch_number > max_epoch:
                max_epoch = epoch_number
                chosen_file = file

    if chosen_file:
        print(f"Found latest epoch model file: {chosen_file} (Epoch {max_epoch})")
        try:
            return chosen_file
        except Exception as e:
            print(f"Error loading epoch model file {chosen_file}: {e}")
            return None
    else:
        print("No valid epoch model files found.")
        return None


def load_and_train_model(input_shape, target_shape, X_train, y_train, X_test, y_test, params, logger):
    """
    Load existing model or create new one, then train it.

    Args:
        input_shape: Shape of input data
        target_shape: Shape of target data
        X_train, y_train: Training data
        X_test, y_test: Test/validation data
        params: Dictionary of training parameters
        logger: Logger instance
    """
    # save all latest the parameters to the pickle file
    params_file = os.path.join(params['results_save_dir'], f"{params['model_name']}_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)

    initial_epoch = 0
    is_train_new_model = False
    if params.get('model_path', None) is not None:
        if params.get('retrain', False) :
            model_path = find_model(params['model_path'])
            if os.path.exists(model_path):
                try:
                    # Load model using keras.models.load_model instead of tf.saved_model.load
                    model = tf.keras.models.load_model(model_path,
                                                       custom_objects={'AntiPersistenceLoss': AntiPersistenceLoss})
                    # model = tf.keras.models.load_model(model_path)
                    logger.info(f"Model loaded from {model_path} and will continue training.")

                    # Get the initial epoch from the model's history if available
                    if hasattr(model, 'history') and model.history is not None:
                        initial_epoch = len(model.history.history['loss'])
                        logger.info(f"Continuing training from epoch {initial_epoch}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    logger.warning("Creating new model instead.")
                    is_train_new_model = True
            else:
                logger.warning(f"Provided model_path {model_path} does not exist. Training a new model.")
                is_train_new_model = True
        else:
            logger.warning(f"re-Train option is false and Training a new model.")
            is_train_new_model = True
    else:
        logger.info("No model path provided. Training a new model.")
        is_train_new_model = True

    # Train model
    if is_train_new_model:
        model = {
            'simple_cnn_lstm': simple_cnn_lstm_model,
            'tcn': tcn_model,
            'tcn_attention': tcn_attention,
            'deep_cnn_lstm': deep_cnn_lstm_model,
            'transformer_positional': transformer_positional,
        }.get(params['model_name'], simple_cnn_lstm_model)(input_shape, target_shape)

    model, history = train_model_full_data(
        model, X_train, y_train, X_test, y_test, params, logger, initial_epoch=initial_epoch,is_train_new_model = is_train_new_model,
    )

    return model, history


