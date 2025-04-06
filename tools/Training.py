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
                              transformer_gaussian, transformer_positional_gaussian, tcn_model, tcn_attention)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # Store the best weights based on monitored metric
        if self.best_weights is None or self.monitor_op(logs[self.monitor], self.best):
            self.best_weights = self.model.get_weights()


@register_keras_serializable()
class MSEComponent(tf.keras.metrics.Metric):
    """
    A custom metric that accumulates the Mean Squared Error over batches.
    """

    def __init__(self, name='mse_component', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_accumulator = self.add_weight(
            name='mse',
            initializer='zeros',
            dtype=self.dtype)
        self.count = self.add_weight(
            name='count',
            initializer='zeros',
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true is cast to the same type as y_pred
        y_true = tf.cast(y_true, y_pred.dtype)

        # Expand dimensions if needed (for multi-step predictions)
        if len(y_true.shape) == 2:
            y_true = tf.expand_dims(y_true, -1)
        if len(y_pred.shape) == 2:
            y_pred = tf.expand_dims(y_pred, -1)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        self.mse_accumulator.assign_add(mse)
        self.count.assign_add(1.0)

    def result(self):
        return self.mse_accumulator / self.count

    def reset_state(self):
        self.mse_accumulator.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        """Creates a metric from its serialized state."""
        return cls(**config)


@register_keras_serializable()
class PersistenceComponent(tf.keras.metrics.Metric):
    """
    A custom metric to measure the 'persistence' of predictions,
    which is a proxy for trend consistency.
    """

    def __init__(self, name='persistence_component', **kwargs):
        super().__init__(name=name, **kwargs)
        self.persistence_accumulator = self.add_weight(
            name='persistence',
            initializer='zeros',
            dtype=self.dtype)
        self.count = self.add_weight(
            name='count',
            initializer='zeros',
            dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, y_pred.dtype)

        if len(y_pred.shape) == 3:  # e.g., (batch, timesteps, 1)
            if len(y_true.shape) == 2:
                y_true = tf.expand_dims(y_true, -1)

        # For multi-step predictions, compute differences between consecutive steps
        if len(y_pred.shape) == 3:
            pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            persistence = tf.reduce_mean(tf.abs(tf.sign(pred_diff)))
        else:  # For single-step, default to a constant
            persistence = tf.constant(1.0, dtype=y_pred.dtype)

        self.persistence_accumulator.assign_add(persistence)
        self.count.assign_add(1.0)

    def result(self):
        return self.persistence_accumulator / self.count

    def reset_state(self):
        self.persistence_accumulator.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        """Creates a metric from its serialized state."""
        return cls(**config)


@register_keras_serializable()
class AntiPersistenceLoss(tf.keras.losses.Loss):
    """
    Custom loss function for forex time series prediction that considers:
    1. Basic MSE for overall prediction accuracy
    2. Direction prediction accuracy (trend following) with smoothing
    3. First prediction accuracy (entry point)
    4. High/Low value matching using smooth approximations
    """

    def __init__(self,shift_penalty_weight=0, prediction_approach='recursive',
                 direction_weight=1.0,
                 first_pred_weight=1.0,
                 highlow_weight=1.0,
                 reduction='sum_over_batch_size',
                 name="anti_persistence_loss"):
        super().__init__(reduction=reduction, name=name)
        self.prediction_approach = prediction_approach
        self.direction_weight = direction_weight
        self.first_pred_weight = first_pred_weight
        self.highlow_weight = highlow_weight

    def call(self, y_true, y_pred):
        if self.prediction_approach == 'multi_steps':
            # Ensure proper shapes (here y_true: (batch, 5, 1) and y_pred: (batch, 5) -> (batch, 5, 1))
            y_true = tf.cast(y_true, y_pred.dtype)
            if len(y_pred.shape) == 2:
                y_pred = tf.expand_dims(y_pred, axis=-1)
            if len(y_true.shape) == 2:
                y_true = tf.expand_dims(y_true, axis=-1)

            # 1. Basic MSE loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # 2. Direction prediction accuracy (smoothed)
            # Apply a simple moving average to reduce the effect of noise.
            # Using a kernel of size 3 (you may adjust as needed).
            kernel = tf.constant([1 / 3.0, 1 / 3.0, 1 / 3.0], dtype=y_true.dtype)
            kernel = tf.reshape(kernel, [3, 1, 1])  # shape: (kernel_size, in_channels, out_channels)
            smoothed_y_true = tf.nn.conv1d(y_true, filters=kernel, stride=1, padding='SAME')
            smoothed_y_pred = tf.nn.conv1d(y_pred, filters=kernel, stride=1, padding='SAME')

            # Compute differences between consecutive (smoothed) time steps.
            y_true_diff = smoothed_y_true[:, 1:, :] - smoothed_y_true[:, :-1, :]
            y_pred_diff = smoothed_y_pred[:, 1:, :] - smoothed_y_pred[:, :-1, :]

            # Instead of a hard sign comparison, use a sigmoid-based penalty.
            # When the product y_true_diff*y_pred_diff is positive (agreeing direction),
            # the sigmoid is near 1 (little penalty); if negative, near 0 (high penalty).
            beta = 5.0  # adjust sensitivity as needed
            directional_agreement = tf.nn.sigmoid(beta * y_true_diff * y_pred_diff)
            direction_loss = tf.reduce_mean(1.0 - directional_agreement)

            # 3. First prediction accuracy (entry point)
            first_pred_loss = tf.reduce_mean(tf.square(y_true[:, 0, :] - y_pred[:, 0, :]))

            # 4. High/Low matching (using smooth approximations)
            # Use a softmax-weighted average to approximate the maximum and minimum in a differentiable way.
            alpha = 10.0  # adjust the temperature as needed
            # Smooth max: weights emphasize larger values.
            smooth_true_max = tf.reduce_sum(y_true * tf.nn.softmax(alpha * y_true, axis=1), axis=1)
            smooth_pred_max = tf.reduce_sum(y_pred * tf.nn.softmax(alpha * y_pred, axis=1), axis=1)
            # Smooth min: weights emphasize smaller values.
            smooth_true_min = tf.reduce_sum(y_true * tf.nn.softmax(-alpha * y_true, axis=1), axis=1)
            smooth_pred_min = tf.reduce_sum(y_pred * tf.nn.softmax(-alpha * y_pred, axis=1), axis=1)

            high_loss = tf.reduce_mean(tf.square(smooth_true_max - smooth_pred_max))
            low_loss = tf.reduce_mean(tf.square(smooth_true_min - smooth_pred_min))
            highlow_loss = (high_loss + low_loss) / 2.0

            # Combine all components with their weights
            total_loss = (
                    mse_loss +
                    self.direction_weight * direction_loss +
                    self.first_pred_weight * first_pred_loss +
                    self.highlow_weight * highlow_loss
            )

            return total_loss
        else:
            # Original recursive approach remains unchanged
            y_true = tf.cast(y_true, y_pred.dtype)
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            pred_current = y_pred[:-1]
            pred_next = y_pred[1:]

            def compute_penalty():
                diff_pred = tf.reduce_mean(tf.square(pred_next - pred_current))
                return tf.exp(-diff_pred)

            def zero_penalty():
                return tf.constant(0.0, dtype=y_pred.dtype)

            penalty = tf.cond(
                tf.greater(tf.shape(y_pred)[0], 1),
                compute_penalty,
                zero_penalty
            )

            return mse_loss + penalty

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "prediction_approach": self.prediction_approach,
            "direction_weight": self.direction_weight,
            "first_pred_weight": self.first_pred_weight,
            "highlow_weight": self.highlow_weight
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


##########################################
# Training Function
##########################################

def train_model_full_data(model, X_train, y_train, X_val, y_val, params, logger, initial_epoch=0,
                          is_train_new_model=True):
    logger.info("Training model with full data loaded into memory")
    # Reshape targets to have an extra channel dimension
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))

    results_dir = params['results_save_dir']

    # Paths for saving different models
    best_model_path = os.path.join(results_dir, params.get('model_name', 'simple_model') + "_best_model.keras")
    final_model_path = os.path.join(results_dir, params.get('model_name', 'simple_model') + "_final_model.keras")
    epoch_model_path = os.path.join(results_dir, params.get('model_name', 'simple_model') + "_epoch_{epoch:02d}.keras")

    # Create custom loss with the specified prediction approach
    custom_loss = AntiPersistenceLoss(
        prediction_approach=params.get('prediction_approach', 'recursive'),
        shift_penalty_weight=params.get('shift_penalty_weight', 0)
    )

    # Create custom checkpoint callback to save the best model based on validation loss
    checkpoint_callback = CustomModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Callback to save the model at each epoch
    epoch_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=epoch_model_path,
        save_weights_only=False,
        #save_freq='epoch',  # Save every epoch
        verbose=1
    )

    callbacks = [
        checkpoint_callback,
        epoch_checkpoint_callback,
        EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    if 'lr_schedule' in params:
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(params['lr_schedule'])
        )

    if is_train_new_model:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get('learning_rate', 0.001),
            clipnorm=params.get('gradient_clip_norm', 1.0)
        )

        metrics = ['mae', 'mse', MSEComponent(), PersistenceComponent()]

        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=metrics
        )

    val_batch_size = params.get('validation_batch_size', params['batch_size'])
    # -------------------  code for the plot
    # for i in range(len(X_train)):
    #     plt.plot(range(len(X_train[i, :])), X_train[i, :])
    #     plt.plot(range(len(X_train[i, :]) + 1, len(X_train[i, :]) + len(y_train[i, :, :]) + 1), y_train[i, :, :])
    #     plt.title('Input Data (X)')
    #     plt.show()
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
            shuffle=False
        )

        # Restore the best weights if available
        if checkpoint_callback.best_weights is not None:
            model.set_weights(checkpoint_callback.best_weights)
            logger.info("Restored best weights from training")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    # Save the training history for later analysis
    history_path = os.path.join(results_dir, params.get('model_name', 'simple_model') + "_history.npy")
    np.save(history_path, history.history)
    logger.info(f"Saved training history to {history_path}")

    # Save the final model after training is complete
    model.save(final_model_path)
    logger.info(f"Saved final model to: {final_model_path}")

    # =======================
    # Activate Visualization:
    # =======================
    # Here we assume X_val[0:1] is a valid sample input for the model.
    sample_input = X_val[0:1] if X_val is not None else None
    visualize_model_training(model, history, sample_input=sample_input,
                             save_dir=os.path.join(results_dir, 'visualizations'))

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
            'transformer_gaussian': transformer_gaussian,
            'transformer_positional_gaussian': transformer_positional_gaussian
        }.get(params['model_name'], simple_cnn_lstm_model)(input_shape, target_shape)

    model, history = train_model_full_data(
        model, X_train, y_train, X_test, y_test, params, logger, initial_epoch=initial_epoch,is_train_new_model = is_train_new_model,
    )

    return model, history


