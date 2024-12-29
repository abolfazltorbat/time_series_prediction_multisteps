import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import os
from tools.BuildModel import (build_simple_cnn_lstm_model,build_simple_cnn_attention_model,
                              build_hybrid_cnn_attention_model,build_hybrid_cnn_transformer_model,
                              build_hybrid_cnn_gp_transformer_model)
def plot_training_history(history, model_name, params, results_dir):
    """
    Plot the training history of the model with academic paper formatting.

    Args:
        history: Training history object containing loss values
        model_name: Name of the model for plot title
        params: Dictionary containing plotting parameters
        results_dir: Directory to save the plot
    """
    logger = logging.getLogger('TimeSeriesModel')
    logger.info(f"Plotting training history for model: {model_name}")

    # Set the style for academic plotting
    plt.style.use('grayscale')

    # Create figure and axis objects with higher DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot training loss
    ax.plot(history.history['loss'],
            label='Training Loss',
            color='#2C3E50',  # Dark blue
            linewidth=1.5,
            marker='o',
            markersize=4,
            markevery=max(len(history.history['loss']) // 20, 1))  # Show markers at intervals

    # Plot validation loss if available
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'],
                label='Validation Loss',
                color='#E74C3C',  # Professional red
                linewidth=1.5,
                marker='s',
                markersize=4,
                markevery=max(len(history.history['val_loss']) // 20, 1))

    # Customize title and labels with LaTeX-style formatting
    ax.set_title(f'Training History: {model_name}',
                 fontsize=14,
                 fontweight='bold',
                 pad=20)

    ax.set_xlabel('Epochs',
                  fontsize=12,
                  fontweight='bold')

    ax.set_ylabel('Loss Value',
                  fontsize=12,
                  fontweight='bold')

    # Format axis ticks
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=10)

    # Customize grid
    ax.grid(True,
            linestyle='--',
            alpha=0.7,
            color='gray')

    # Enhanced legend
    ax.legend(loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=10,
              bbox_to_anchor=(0.99, 0.99))

    # Scientific notation for y-axis if values are very small
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Add minor ticks
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':',
            alpha=0.3, color='gray')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure with high quality
    save_path = os.path.join(results_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path,
                format='png',
                bbox_inches='tight',
                pad_inches=0.1,
                metadata={'Creator': 'Training History Plot'})

    if params['is_plot']:
        plt.show()
    plt.close()


def train_model_full_data(model, X_train, y_train, X_val, y_val, params, logger, initial_epoch=0):
    """
    Train the model on full data loaded into memory with support for continuing training.

    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: Dictionary of training parameters
        logger: Logger instance
        initial_epoch: Epoch to start training from (useful for continued training)
    """
    logger.info("Training model with full data loaded into memory")
    results_dir = params['results_save_dir']
    model_checkpoint_path = os.path.join(results_dir, "best_model_full_data.keras")

    checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )

    history = model.fit(
        X_train,
        y_train,
        initial_epoch=initial_epoch,  # Start from this epoch
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Save training plots
    plot_training_history(history, 'Full_Data_Model', params, results_dir)
    plot_model(
        model,
        to_file=os.path.join(results_dir, f"Full_Data_Model_architecture.png"),
        show_shapes=True,
        show_layer_names=True,
        dpi=params['dpi']
    )

    return model, history


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
    initial_epoch = 0
    is_train_new_model = False
    if params.get('model_path', None) is not None:
        model_path = params['model_path']
        if os.path.exists(model_path):
            try:
                # Load model using keras.models.load_model instead of tf.saved_model.load
                model = tf.keras.models.load_model(model_path)
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
        logger.info("No model path provided. Training a new model.")
        is_train_new_model = True

    # Train model

    if is_train_new_model:
        model = build_simple_cnn_lstm_model(input_shape, target_shape)
        # model = build_simple_cnn_attention_model(input_shape, target_shape)
        # model = build_hybrid_cnn_attention_model(input_shape, target_shape, params, logger)
        # model = build_hybrid_cnn_transformer_model(input_shape, target_shape, params, logger)
        # model = build_hybrid_cnn_gp_transformer_model(input_shape, target_shape, params, logger)

    model, history = train_model_full_data(
        model, X_train, y_train, X_test, y_test, params, logger, initial_epoch=initial_epoch
    )

    return model, history