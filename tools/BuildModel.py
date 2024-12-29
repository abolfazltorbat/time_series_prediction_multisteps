import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Dense, LayerNormalization, Add, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, Layer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import numpy as np

import logging



def get_optimizer(name, learning_rate):
    """Get the optimizer by name."""
    if name.lower() == 'adam':
        return Adam(learning_rate=learning_rate)
    elif name.lower() == 'sgd':
        return SGD(learning_rate=learning_rate)
    elif name.lower() == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        return Adam(learning_rate=learning_rate)

def build_simple_cnn_lstm_model(input_shape, target_shape, lstm_units=64, dff=128, cnn_filters=64, kernel_size=3,
                                dropout_rate=0.00):
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Building Hybrid CNN-LSTM model")

    inputs = Input(shape=(input_shape[0], 1))
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(cnn_output)
    cnn_output = Dropout(dropout_rate)(cnn_output)

    # Replace attention with LSTM
    lstm_output = LSTM(lstm_units, return_sequences=True)(cnn_output)
    lstm_output = Dropout(dropout_rate)(lstm_output)
    lstm_output = LayerNormalization(epsilon=1e-6)(Add()([cnn_output, lstm_output]))

    ffn_output = Dense(dff, activation='relu')(lstm_output)
    ffn_output = Dense(cnn_filters)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(Add()([lstm_output, ffn_output]))

    flat = Flatten()(ffn_output)
    outputs = Dense(target_shape[-1])(flat)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_simple_cnn_attention_model(input_shape, target_shape, num_heads=4, dff=128, cnn_filters=64, kernel_size=3,
                                     dropout_rate=0.01):
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Building Hybrid CNN-Attention model")
    inputs = Input(shape=(input_shape[0], 1))
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    cnn_output = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(cnn_output)
    cnn_output = Dropout(dropout_rate)(cnn_output)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=cnn_filters)(cnn_output, cnn_output)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(Add()([cnn_output, attn_output]))
    ffn_output = Dense(dff, activation='relu')(attn_output)
    ffn_output = Dense(cnn_filters)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(Add()([attn_output, ffn_output]))
    flat = Flatten()(ffn_output)
    outputs = Dense(target_shape[-1])(flat)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_hybrid_cnn_attention_model(input_shape, target_shape, params, logger):
    """Build the hybrid CNN-Attention model."""
    logger.info("Building Enhanced Hybrid CNN-Attention model")
    inputs = Input(shape=(input_shape[0], 1))
    x = inputs
    for _ in range(params.get('num_cnn_layers', 4)):
        x = Conv1D(
            filters=params.get('cnn_filters', 64),
            kernel_size=params.get('kernel_size', 3),
            activation='relu',
            padding='same'
        )(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)
    for _ in range(params.get('num_attention_layers', 2)):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=params.get('num_heads', 8),
            key_dim=params.get('key_dim', 64)
        )(x, x)
        attn_output = Dropout(params.get('dropout_rate', 0.2))(attn_output)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))
    x = Flatten()(x)
    for units in params.get('dense_units', [256, 128]):
        x = Dense(units, activation='relu')(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)
    outputs = Dense(target_shape[-1])(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer_name = params.get('optimizer', 'adam')
    learning_rate = params.get('learning_rate', 0.001)
    optimizer = get_optimizer(optimizer_name, learning_rate)
    loss_function = params.get('loss_function', 'mse')
    model.compile(optimizer=optimizer, loss=loss_function)
    return model


def build_hybrid_cnn_transformer_model(input_shape, target_shape, params, logger):
    """Build an advanced model for time series data with state-of-the-art approaches."""
    logger.info("Building Advanced Time Series Model with Transformer and Positional Encoding")

    inputs = Input(shape=(input_shape[0], 1))
    x = inputs

    # Convolutional layers
    for _ in range(params.get('num_cnn_layers', 4)):
        x = Conv1D(
            filters=params.get('cnn_filters', 64),
            kernel_size=params.get('kernel_size', 3),
            activation='relu',
            padding='same'
        )(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Project to desired dimension for the transformer
    d_model = params.get('d_model', 128)
    x = Dense(d_model)(x)

    # Positional Encoding
    x = PositionalEncoding(input_shape[0], d_model)(x)

    # Transformer Encoder layers
    for _ in range(params.get('num_transformer_layers', 2)):
        x = EncoderLayer(
            d_model=d_model,
            num_heads=params.get('num_heads', 8),
            dff=params.get('dff', 512),
            rate=params.get('dropout_rate', 0.1)
        )(x, training=True)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Dense layers
    for units in params.get('dense_units', [256, 128]):
        x = Dense(units, activation='relu')(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

    outputs = Dense(target_shape[-1])(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer_name = params.get('optimizer', 'adam')
    learning_rate = params.get('learning_rate', 0.001)
    optimizer = get_optimizer(optimizer_name, learning_rate)
    loss_function = params.get('loss_function', 'mse')
    model.compile(optimizer=optimizer, loss=loss_function)

    return model



# deeper model
class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # Apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        return x


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(
            query=x, key=x, value=x, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

import tensorflow_probability as tfp

# Define the GP Kernel Attention Layer
class GPKernelAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(GPKernelAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # Define the query, key, and value dense layers
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

        # Initialize the GP kernel (RBF kernel)
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=1.0)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Compute attention weights using the GP kernel
        # Reshape for batch computation
        q = tf.reshape(q, (batch_size * self.num_heads, -1, self.depth))
        k = tf.reshape(k, (batch_size * self.num_heads, -1, self.depth))
        v = tf.reshape(v, (batch_size * self.num_heads, -1, self.depth))

        # Compute pairwise squared distances
        dists = tf.reduce_sum(q**2, axis=2, keepdims=True) - 2 * tf.matmul(q, k, transpose_b=True) + tf.transpose(
            tf.reduce_sum(k**2, axis=2, keepdims=True), perm=[0, 2, 1]
        )

        # Apply the RBF kernel
        attention_weights = tf.exp(-0.5 * dists / self.kernel.length_scale**2)

        # Apply mask if provided
        if mask is not None:
            attention_weights += (mask * -1e9)

        # Normalize the attention weights
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        # Compute the attention output
        output = tf.matmul(attention_weights, v)

        # Restore the original shape
        output = tf.reshape(output, (batch_size, self.num_heads, -1, self.depth))
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        # Final linear layer
        output = self.dense(output)

        return output

# Modify the EncoderLayer to use GPKernelAttention
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = GPKernelAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(v=x, k=x, q=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Build the hybrid model with the GP-enhanced Transformer
def build_hybrid_cnn_gp_transformer_model(input_shape, target_shape, params, logger):
    logger.info("Building Advanced Time Series Model with Transformer and GP-enhanced Attention")

    inputs = Input(shape=(input_shape[0], 1))
    x = inputs

    # Convolutional layers
    for _ in range(params.get('num_cnn_layers', 4)):
        x = Conv1D(
            filters=params.get('cnn_filters', 64),
            kernel_size=params.get('kernel_size', 3),
            activation='relu',
            padding='same'
        )(x)
    x = Dropout(params.get('dropout_rate', 0.2))(x)

    # Project to desired dimension for the transformer
    d_model = params.get('d_model', 128)
    x = Dense(d_model)(x)

    # Positional Encoding
    x = PositionalEncoding(input_shape[0], d_model)(x)

    # Transformer Encoder layers with GP-enhanced attention
    for _ in range(params.get('num_transformer_layers', 2)):
        x = EncoderLayer(
            d_model=d_model,
            num_heads=params.get('num_heads', 8),
            dff=params.get('dff', 512),
            rate=params.get('dropout_rate', 0.1)
        )(x, training=True)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    for units in params.get('dense_units', [256, 128]):
        x = Dense(units, activation='relu')(x)
        x = Dropout(params.get('dropout_rate', 0.2))(x)

    outputs = Dense(target_shape[-1])(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer_name = params.get('optimizer', 'adam')
    learning_rate = params.get('learning_rate', 0.001)
    optimizer = get_optimizer(optimizer_name, learning_rate)
    loss_function = params.get('loss_function', 'mse')
    model.compile(optimizer=optimizer, loss=loss_function)

    return model

