import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Conv1D,
    Dropout,
    Flatten,
    Activation,
    Dense,
    TimeDistributed,
    LayerNormalization,
    Add,
    RepeatVector,
    LSTM,
    GRU,
    Attention,
    GlobalAveragePooling1D,
    Layer,
    MultiHeadAttention,
    BatchNormalization,
    Softmax,
    Multiply,
    GlobalMaxPooling1D,
    Bidirectional
)
from tensorflow.keras.models import Model, Sequential
from keras.saving import register_keras_serializable

###############################################################################
# CNN + LSTM MODELS
###############################################################################

def simple_cnn_lstm_model(
    inp_shape,
    tgt_shape,
    lstm_units=64,
    dff=128,
    cnn_filters=64,
    kernel_size=3,
    model_name = 'simple_cnn_lstm_model',
    dropout_rate=0.3
):
    """
    Simple CNN + LSTM model.
    """
    inputs = Input(shape=(inp_shape[0], 1))
    c = Conv1D(cnn_filters, kernel_size, activation='relu', padding='same')(inputs)
    c = Conv1D(cnn_filters, kernel_size, activation='relu', padding='same')(c)
    c = Dropout(dropout_rate)(c)

    l = LSTM(lstm_units, return_sequences=True)(c)
    l = Dropout(dropout_rate)(l)
    l = LayerNormalization(epsilon=1e-6)(Add()([c, l]))

    f = Dense(dff, activation='relu')(l)
    f = Dense(cnn_filters)(f)
    f = Dropout(dropout_rate)(f)
    f = LayerNormalization(epsilon=1e-6)(Add()([l, f]))

    x = Flatten()(f)
    outputs = Dense(tgt_shape[-1])(x)

    model = Model(inputs, outputs,name=model_name)
    return model



def deep_cnn_lstm_model(
        inp_shape,
        tgt_shape,
        lstm_units=64,
        dff=128,
        cnn_filters=64,
        kernel_size=3,
        model_name='deep_cnn_lstm_model',
        dropout_rate=0.1
):
    # Input layer
    inputs = Input(shape=(inp_shape[0], 1))

    # Multi-scale CNN blocks
    c1 = Conv1D(cnn_filters, 3, activation='relu', padding='same')(inputs)
    c2 = Conv1D(cnn_filters, 5, activation='relu', padding='same')(inputs)
    c3 = Conv1D(cnn_filters, 7, activation='relu', padding='same')(inputs)

    # Combine CNN features
    c = Concatenate()([c1, c2, c3])
    c = Conv1D(cnn_filters * 2, 3, activation='relu', padding='same')(c)
    c = BatchNormalization()(c)
    c = Dropout(dropout_rate)(c)

    # Bi-directional LSTM layers
    l = Bidirectional(LSTM(lstm_units, return_sequences=True))(c)
    l = Dropout(dropout_rate)(l)
    l = LayerNormalization(epsilon=1e-6)(l)

    # Second LSTM with skip connection
    l2 = Bidirectional(LSTM(lstm_units , return_sequences=True))(l)
    l2 = Dropout(dropout_rate)(l2)
    l = Add()([l, l2])  # Skip connection
    l = LayerNormalization(epsilon=1e-6)(l)

    # Attention mechanism
    a = Attention()([l, l])

    # Temporal feature extraction before final prediction
    pooled_features = GlobalAveragePooling1D()(a)

    # Use LSTM decoder for sequence generation
    decoder = RepeatVector(tgt_shape[0])(pooled_features)
    decoder = LSTM(lstm_units, return_sequences=True)(decoder)

    # Output layer (sequence to sequence)
    outputs = TimeDistributed(Dense(1))(decoder)

    model = Model(inputs, outputs, name='enhanced_cnn_lstm')
    return model

# #############################################################################
# -----------TCN- attention model
##############################################################################
def tcn_attention(
        inp_shape,
        tgt_shape,
        num_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        dropout_rate=0.1,
        model_name="tcn_attention"
):
    """
    A simple Temporal Convolutional Network (TCN) model for time series forecasting.

    Args:
        inp_shape (tuple): Shape of the input sequence, e.g. (seq_length,).
        tgt_shape (tuple): Shape of the output, e.g. (num_targets,).
        num_filters (int): Number of convolutional filters in each TCN layer.
        kernel_size (int): Size of the convolution kernel.
        dilations (list): Dilation rates for each residual block.
        dropout_rate (float): Dropout probability applied inside TCN blocks.
        model_name (str): Name of the resulting Keras model.

    Returns:
        A Keras Model object.
    """
    # Modified input shape to match the CNN-LSTM model structure
    inputs = Input(shape=(inp_shape[0], 1))  # Changed to match your input structure

    x = inputs
    for dilation_rate in dilations:
        # -- Residual Block --
        shortcut = x

        # 1) First dilated causal conv
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu"
        )(x)
        x = Dropout(dropout_rate)(x)

        # 2) Second dilated causal conv
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu"
        )(x)
        x = Dropout(dropout_rate)(x)

        # -- Match channels for the shortcut if needed --
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv1D(filters=num_filters, kernel_size=1, padding="same")(shortcut)

        # -- Residual/skip connection --
        x = Add()([shortcut, x])
        x = LayerNormalization(epsilon=1e-6)(x)

    # -- Global pooling + Dense for final output --
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(tgt_shape[-1], activation="linear")(x)

    model = Model(inputs, outputs, name=model_name)
    return model


def tcn_model(
        inp_shape,
        tgt_shape,
        num_filters=128,  # Increased from 64
        kernel_size=5,  # Increased from 3
        dilations=[1, 2, 4, 8, 16, 32],  # Extended dilations for longer-term patterns
        dropout_rate=0.1,  # Increased for better regularization
        l2_reg=1e-6,  # Added L2 regularization
        model_name="tcn_model_forex"
):
    """
    Enhanced TCN model optimized for forex 1-minute data prediction.

    Args:
        inp_shape (tuple): Shape of the input sequence (seq_length,)
        tgt_shape (tuple): Shape of the output (num_targets,)
        num_filters (int): Number of convolutional filters in each TCN layer
        kernel_size (int): Size of the convolution kernel
        dilations (list): Dilation rates for each residual block
        dropout_rate (float): Dropout probability
        l2_reg (float): L2 regularization factor
        model_name (str): Name of the model

    Returns:
        A Keras Model object optimized for forex prediction
    """
    inputs = Input(shape=(inp_shape[0], 1))

    # Initial normalization layer
    x = LayerNormalization(epsilon=1e-6)(inputs)

    # Multiple stacks of TCN blocks with different dilation rates
    skip_connections = []

    for dilation_rate in dilations:
        # -- Residual Block --
        shortcut = x

        # 1) First dilated causal conv with L2 regularization
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout_rate)(x)

        # 2) Second dilated causal conv with L2 regularization
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout_rate)(x)

        # -- Match channels for the shortcut if needed --
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv1D(
                filters=num_filters,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg)
            )(shortcut)

        # -- Residual connection --
        x = Add()([shortcut, x])

        # Store skip connection
        skip_connections.append(x)

    # Combine skip connections
    if len(skip_connections) > 1:
        x = Add()(skip_connections)
    else:
        x = skip_connections[0]

    # Final processing
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Activation("relu")(x)

    # Attention mechanism for capturing important temporal patterns
    attention = Dense(num_filters, use_bias=False)(x)
    attention = Activation("tanh")(attention)
    attention = Dense(1, use_bias=False)(attention)
    attention = Activation("softmax")(attention)
    x = Multiply()([x, attention])

    # Global pooling
    x = GlobalAveragePooling1D()(x)

    # Final dense layers with dropout
    x = Dense(
        num_filters,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Output layer
    outputs = Dense(
        tgt_shape[-1],
        activation="linear",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    model = Model(inputs, outputs, name=model_name)
    return model
###############################################################################
# 2) TRANSFORMER COMPONENTS
###############################################################################

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, LayerNormalization, Conv1D, Reshape, Input,
    MultiHeadAttention # Using built-in MHA might be simpler, but keeping custom for now
)
from tensorflow.keras.models import Model
import math # For GELU approximation if needed, though tf.nn.gelu is preferred

# --- Helper Functions (Assuming GELU is needed, tf.nn.gelu is recommended) ---
# Using tf.nn.gelu is generally better
def gelu(x):
    """
    Gaussian Error Linear Unit activation function.
    Using tf.nn.gelu(approximate=True) might be more efficient.
    """
    # return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return tf.nn.gelu(x) # Use TensorFlow's built-in GELU

# --- Custom Layers (Revised RoPE and others) ---

@tf.keras.utils.register_keras_serializable()
class RotaryPositionalEmbedding(Layer):
    """
    Implements Rotary Position Embedding (RoPE).
    Calculations moved to `call` to handle potentially dynamic sequence lengths.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        if dim % 2 != 0:
            raise ValueError(f"Dimension ({dim}) must be even for Rotary Positional Embedding.")
        self.dim = dim
        self.inv_freq = None # Initialize inv_freq

    def build(self, input_shape):
        # Calculate inverse frequencies - depends only on dim
        # Shape: (dim / 2,)
        self.inv_freq = 1.0 / (10000 ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        super().build(input_shape) # Ensure superclass build is called

    def call(self, x):
        # x shape: [batch_size, seq_len, dim]
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        dtype = x.dtype # Use the dtype of the input tensor

        # Calculate positional information based on current sequence length
        # position shape: (seq_len,)
        position = tf.range(seq_len, dtype=dtype)
        # sincos_freq shape: (seq_len, dim / 2)
        # Ensure inv_freq uses the same dtype
        sincos_freq = tf.einsum('i,j->ij', position, tf.cast(self.inv_freq, dtype=dtype))
        # sin shape: (seq_len, dim / 2)
        # cos shape: (seq_len, dim / 2)
        sin = tf.sin(sincos_freq)
        cos = tf.cos(sincos_freq)

        # Expand sin and cos for broadcasting: [1, seq_len, dim / 2]
        sin = sin[tf.newaxis, :, :]
        cos = cos[tf.newaxis, :, :]

        # Split the last dimension to prepare for rotation
        # x_reshape shape: [batch_size, seq_len, dim/2, 2]
        # The feature dimension (self.dim) must be the last dimension for reshape
        x_reshape = tf.reshape(x, [batch_size, seq_len, self.dim // 2, 2])
        # x1, x2 shapes: [batch_size, seq_len, dim/2]
        x1 = x_reshape[..., 0]
        x2 = x_reshape[..., 1]

        # Apply rotation using the formula:
        # rotated_x1 = x1*cos - x2*sin
        # rotated_x2 = x1*sin + x2*cos
        # Broadcasting happens here: [batch_size, seq_len, dim/2] * [1, seq_len, dim/2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Concatenate the rotated values
        # rotated shape: [batch_size, seq_len, dim/2, 2]
        rotated = tf.stack([rotated_x1, rotated_x2], axis=-1)
        # Reshape back to original dim: [batch_size, seq_len, dim]
        rotated = tf.reshape(rotated, [batch_size, seq_len, self.dim])

        return rotated

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEmbedding(Layer):
    """
    Implements Patch Embedding for time series data.
    """
    def __init__(self, d_model, patch_size=24, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.patch_size = patch_size
        # Define layers in init or build
        self.projection = Conv1D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name='patch_projection' # Add name for clarity
        )
        # The target shape for Reshape should not include the batch dimension
        self.flatten = Reshape((-1, d_model), name='patch_flatten') # Target shape: (num_patches, d_model)

    def build(self, input_shape):
        # You can also define layers here if they depend on input_shape
        # self.projection = Conv1D(...)
        # self.flatten = Reshape(...)
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch_size, seq_len, features) e.g., (None, 1440, 1)
        x = self.projection(x) # (None, num_patches, d_model) e.g., (None, 60, 128)
        x = self.flatten(x)     # (None, num_patches, d_model) - Reshape might be redundant if Conv1D output is already correct
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "patch_size": self.patch_size
        })
        return config

# Keep custom MHA and CrossAttention as they are, assuming they work as intended
# Or replace with tf.keras.layers.MultiHeadAttention for simplicity/robustness
@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    """
    Custom Multi-head Self-Attention layer.
    Consider using tf.keras.layers.MultiHeadAttention.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.depth = d_model // num_heads

        self.wq = Dense(d_model, name='q_dense')
        self.wk = Dense(d_model, name='k_dense')
        self.wv = Dense(d_model, name='v_dense')
        self.dropout_layer = Dropout(dropout_rate) # Renamed to avoid conflict
        self.dense = Dense(d_model, name='out_dense')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, seq_len, d_model)
        batch_size = tf.shape(inputs)[0]

        q = self.wq(inputs)  # (batch_size, seq_len_q, d_model)
        k = self.wk(inputs)  # (batch_size, seq_len_k, d_model)
        v = self.wv(inputs)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        # matmul_qk shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale matmul_qk
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if provided
        if mask is not None:
            # The mask needs to broadcast correctly, e.g., (batch_size, 1, 1, seq_len_k)
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout_layer(attention_weights) # Use the dropout layer instance

        # output shape: (batch_size, num_heads, seq_len_q, depth)
        output = tf.matmul(attention_weights, v)

        # Transpose back: (batch_size, seq_len_q, num_heads, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])

        # Concatenate heads: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        # Apply final linear layer: (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

@tf.keras.utils.register_keras_serializable()
class CrossAttention(Layer):
    """
    Custom Cross-Attention layer.
    Consider using tf.keras.layers.MultiHeadAttention.
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.depth = d_model // num_heads

        self.wq = Dense(d_model, name='q_dense')
        self.wk = Dense(d_model, name='k_dense')
        self.wv = Dense(d_model, name='v_dense')
        self.dropout_layer = Dropout(dropout_rate) # Renamed
        self.dense = Dense(d_model, name='out_dense')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    def call(self, queries, keys, values, mask=None):
        # queries shape: (batch_size, seq_len_q, d_model)
        # keys shape: (batch_size, seq_len_k, d_model)
        # values shape: (batch_size, seq_len_v, d_model) - seq_len_k == seq_len_v typically
        batch_size = tf.shape(queries)[0]

        q = self.wq(queries)  # (batch_size, seq_len_q, d_model)
        k = self.wk(keys)    # (batch_size, seq_len_k, d_model)
        v = self.wv(values)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True) # (batch_size, num_heads, seq_len_q, seq_len_k)

        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = self.dropout_layer(attention_weights) # Use dropout layer instance

        output = tf.matmul(attention_weights, v) # (batch_size, num_heads, seq_len_q, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

@tf.keras.utils.register_keras_serializable()
class ClassTokenLayer(Layer):
    """ Adds a learnable class token to the input sequence. """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, d_model)
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_model), # Shape (1, 1, d_model) for easy concatenation
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, d_model)
        batch_size = tf.shape(inputs)[0]
        # Repeat cls_token along batch dimension
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0) # (batch_size, 1, d_model)
        # Concatenate along the sequence dimension (axis=1)
        return tf.concat([cls_tokens, inputs], axis=1) # (batch_size, seq_len + 1, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

@tf.keras.utils.register_keras_serializable()
class DecoderInitializer(Layer):
    """ Initializes decoder inputs, e.g., with zeros. """
    def __init__(self, d_model, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.seq_length = seq_length # Target sequence length for decoder input

    def build(self, input_shape):
        # Doesn't necessarily depend on input_shape if just creating zeros
        super().build(input_shape)

    def call(self, reference_tensor):
        # Use a reference tensor (like encoder output) to get batch size and dtype
        batch_size = tf.shape(reference_tensor)[0]
        dtype = reference_tensor.dtype
        # Return zeros of shape (batch_size, target_seq_length, d_model)
        return tf.zeros([batch_size, self.seq_length, self.d_model], dtype=dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_length": self.seq_length
        })
        return config

# --- Feed Forward Network ---
def feed_forward_network(d_model, hidden_dim, dropout_rate=0.1, name="feed_forward_network"):
    """ Position-wise Feed-Forward Network with GELU activation. """
    return tf.keras.Sequential([
        Dense(hidden_dim, activation=gelu, name='ffn_dense_1'),
        Dropout(dropout_rate),
        Dense(d_model, name='ffn_dense_2'),
        Dropout(dropout_rate)
    ], name=name)

# --- Masking Functions (Unchanged) ---

def create_padding_mask(seq):
    """ Creates mask for padding tokens (assuming 0 is padding). """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """ Creates a look-ahead mask for decoder self-attention. """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

@tf.keras.utils.register_keras_serializable()
class LookAheadMaskLayer(Layer):
    """Creates a look-ahead mask for decoder self-attention based on input shape."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]  # Get sequence length dynamically
        mask = create_look_ahead_mask(seq_len)  # Create look-ahead mask
        # Expand mask dimensions to (batch_size, 1, seq_len, seq_len) for attention
        mask = mask[tf.newaxis, tf.newaxis, :, :]  # Shape: (1, 1, seq_len, seq_len)
        return mask

    def get_config(self):
        return super().get_config()

# --- Transformer Blocks (Revised RoPE Application) ---
def transformer_encoder_block(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1, name="encoder_block"):
    """
    Transformer encoder block with pre-RoPE application.
    """
    with tf.name_scope(name):
        # Apply rotary positional embedding with a unique name
        rotated_inputs = RotaryPositionalEmbedding(d_model, name=f"rope_{name}")(inputs)

        # Multi-head self-attention on rotated inputs with a unique name
        attn_output = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"self_attention_{name}"
        )(rotated_inputs)

        attn_output = Dropout(dropout_rate)(attn_output)
        # Residual connection from *original* inputs with a unique name
        out1 = LayerNormalization(epsilon=1e-6, name=f"layernorm_1_{name}")(inputs + attn_output)

        # Feed Forward Network
        ffn_output = feed_forward_network(d_model, ff_dim, dropout_rate, name=f"ffn_{name}")(out1)

        # Residual connection with a unique name
        out2 = LayerNormalization(epsilon=1e-6, name=f"layernorm_2_{name}")(out1 + ffn_output)
        return out2

def transformer_decoder_block(inputs, encoder_outputs, d_model, num_heads, ff_dim, dropout_rate=0.1, name="decoder_block"):
    """
    Transformer decoder block with pre-RoPE for self-attention.
    """
    with tf.name_scope(name):
        # Apply rotary positional embedding with a unique name
        rotated_inputs = RotaryPositionalEmbedding(d_model, name=f"rope_{name}")(inputs)

        # Create look-ahead mask using the custom layer
        look_ahead_mask = LookAheadMaskLayer(name=f"look_ahead_mask_{name}")(rotated_inputs)

        # Masked Multi-head self-attention on rotated inputs with a unique name
        self_attn_output = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"masked_self_attention_{name}"
        )(rotated_inputs, mask=look_ahead_mask)

        self_attn_output = Dropout(dropout_rate)(self_attn_output)
        # Residual connection from *original* decoder inputs with a unique name
        out1 = LayerNormalization(epsilon=1e-6, name=f"layernorm_1_{name}")(inputs + self_attn_output)

        # Cross attention with *original* (non-rotated) encoder outputs
        cross_attn_output = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"cross_attention_{name}"
        )(queries=out1, keys=encoder_outputs, values=encoder_outputs)

        cross_attn_output = Dropout(dropout_rate)(cross_attn_output)
        # Residual connection with a unique name
        out2 = LayerNormalization(epsilon=1e-6, name=f"layernorm_2_{name}")(out1 + cross_attn_output)

        # Feed Forward Network
        ffn_output = feed_forward_network(d_model, ff_dim, dropout_rate, name=f"ffn_{name}")(out2)

        # Residual connection with a unique name
        out3 = LayerNormalization(epsilon=1e-6, name=f"layernorm_3_{name}")(out2 + ffn_output)
        return out3

@tf.keras.utils.register_keras_serializable()
class SliceToTargetLength(Layer):
    """Slices the input tensor to the specified target length along the time dimension."""
    def __init__(self, target_length, **kwargs):
        super().__init__(**kwargs)
        self.target_length = target_length

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        # Slice to (batch_size, target_length, features)
        return inputs[:, :self.target_length, :]

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, seq_len, features)
        return (input_shape[0], self.target_length, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length})
        return config

# --- Model Definition (Revised) ---
def transformer_positional(
        inp_shape, # e.g., (1440, 1)
        tgt_shape, # e.g., (90, 1)
        model_name='transformer_positional',
        d_model=128,
        num_heads=8,
        ff_dim=512,
        dropout_rate=0.1,
        num_encoder_layers=6,
        num_decoder_layers=4,
        patch_size=24
):
    """
    Transformer architecture using PatchEmbedding and Rotary Positional Embedding.

    Args:
        inp_shape: Shape of input time series (sequence_length, features) -> (1440, 1)
        tgt_shape: Shape of target time series (target_length, features) -> (90, 1)
        model_name: Name of the model
        d_model: Dimension of the model embeddings and layers
        num_heads: Number of attention heads
        ff_dim: Hidden dimension of the feed-forward network
        dropout_rate: Dropout rate
        num_encoder_layers: Number of encoder blocks
        num_decoder_layers: Number of decoder blocks
        patch_size: Size of patches for input embedding
    Returns:
        Compiled Keras Transformer model.
    """
    # Input Layer
    # Expecting shape (batch_size, sequence_length, features)
    inputs = Input(shape=(inp_shape[0], 1), name="input_time_series") # Use inp_shape[1] for features

    # --- Encoder ---
    # 1. Patch Embedding
    # Input: (None, 1440, 1) -> Output: (None, 1440/24, d_model) = (None, 60, 128)
    encoder_x = PatchEmbedding(d_model, patch_size=patch_size, name="patch_embedding")(inputs)

    # 2. Add Class Token (Optional, common in ViT style models)
    # Input: (None, 60, 128) -> Output: (None, 61, 128)
    encoder_x = ClassTokenLayer(d_model, name="class_token")(encoder_x)

    # 3. Encoder Stack (with pre-RoPE)
    # Input: (None, 61, 128) -> Output: (None, 61, 128)
    encoder_outputs = encoder_x
    for i in range(num_encoder_layers):
        encoder_outputs = transformer_encoder_block(
            encoder_outputs, d_model, num_heads, ff_dim, dropout_rate, name=f"encoder_block_{i + 1}")

    # Separate Class token and Sequence output if ClassToken was used
    # class_token_output = encoder_outputs[:, 0, :] # Shape: (None, d_model)
    encoder_sequence_output = encoder_outputs[:, 1:, :] # Shape: (None, 60, 128) - Use this for decoder

    # --- Decoder ---
    # 1. Initialize Decoder Input
    # Calculate target sequence length in terms of patches/steps for the decoder
    # The decoder needs to predict `tgt_shape[0]` steps. How this relates to patches needs careful thought.
    # If the decoder operates on the same patch level as the encoder, the target length might be `tgt_shape[0] // patch_size`.
    # If the decoder directly predicts the final time steps, its input length might be `tgt_shape[0]`.
    # The original code calculated prediction_length based on patches. Let's stick to that for now.
    # It implies the decoder outputs patch representations first.
    # --- Decoder ---
    # 1. Initialize Decoder Input
    prediction_length_patches = tgt_shape[0] // patch_size + (
        1 if tgt_shape[0] % patch_size > 0 else 0)  # 90 -> 4 patches

    decoder_inputs = DecoderInitializer(
        d_model, prediction_length_patches, name="decoder_initializer"
    )(encoder_sequence_output)  # Output shape: (None, 4, 128)

    # 2. Decoder Stack (with pre-RoPE for self-attn, cross-attn with encoder seq)
    decoder_outputs = decoder_inputs
    for i in range(num_decoder_layers):
        decoder_outputs = transformer_decoder_block(
            decoder_outputs,
            encoder_sequence_output,
            d_model,
            num_heads,
            ff_dim,
            dropout_rate,
            name=f"decoder_block_{i + 1}"
        )
    # --- Final Output ---
    # Project decoder outputs (patch representations) to the patch size
    # Input: (None, 4, 128) -> Output: (None, 4, patch_size) = (None, 4, 24)
    patched_outputs = Dense(patch_size, name="patch_output_projection")(decoder_outputs)

    # Reshape patches back into a continuous time series
    # Input: (None, 4, 24) -> Output: (None, 4 * 24, 1) = (None, 96, 1)
    # Assuming the target feature dimension is 1
    reshaped_outputs = Reshape((-1, 1), name="reshape_to_time_series")(patched_outputs)

    # Slice to the exact target length (e.g., 90)
    # Input: (None, 96, 1) -> Output: (None, 90, 1)
    # Using a Lambda layer for slicing
    final_outputs = SliceToTargetLength(
        target_length=tgt_shape[0],
        name="slice_to_target_length"
    )(reshaped_outputs)

    # Create model
    model = Model(inputs=inputs, outputs=final_outputs, name=model_name)
    return model

# Example Usage (define shapes first)
# INPUT_SHAPE = (1440, 1) # (Sequence Length, Features)
# TARGET_SHAPE = (90, 1) # (Target Length, Features)

# model = transformer_positional(
#     inp_shape=INPUT_SHAPE,
#     tgt_shape=TARGET_SHAPE,
#     d_model=128,
#     num_heads=8,
#     ff_dim=512,
#     dropout_rate=0.1,
#     num_encoder_layers=6,
#     num_decoder_layers=4,
#     patch_size=24
# )

# model.summary()
# tf.keras.utils.plot_model(model, to_file='transformer_positional.png', show_shapes=True)

