import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
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

class PositionalEncoding(Layer):
    """
    Enhanced Positional Encoding layer that combines absolute and relative position information.
    This implementation extends the standard transformer positional encoding by adding:
    1. Relative position awareness
    2. Enhanced angle calculations
    3. Customizable scaling factors
    4. Support for longer sequences

    Args:
        position (int): Maximum sequence length to encode
        d_model (int): Dimensionality of the model/embedding space
        max_relative_position (int, optional): Maximum distance to consider for relative positions.
            Defaults to 64.
        scaling_factor (float, optional): Scaling factor for positional encodings.
            Defaults to 1.0.
    """

    def __init__(self, position, d_model, max_relative_position=64, scaling_factor=1.0, **kwargs):
        # Accept **kwargs and pass to super().__init__
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.scaling_factor = scaling_factor

        # Initialize absolute positional encodings
        self.abs_pos_encoding = self.create_absolute_positional_encoding()
        # Initialize relative positional encodings
        self.rel_pos_encoding = self.create_relative_positional_encoding()

    def get_angles(self, pos, i, d_model):
        """
        Calculate the angles for positional encoding using an enhanced formula.

        Args:
            pos (np.array): Position indices
            i (np.array): Dimension indices
            d_model (int): Model dimension size

        Returns:
            np.array: Calculated angles for the positional encoding
        """
        # Enhanced angle calculation with better numerical stability
        # Using log space to prevent numerical overflow for long sequences
        exponent = (2 * (i // 2)) / np.float32(d_model)
        denominator = np.power(10000, exponent)
        angles = pos / denominator
        return angles

    def create_absolute_positional_encoding(self):
        """
        Creates the absolute positional encoding matrix.
        Uses sinusoidal functions to create position-dependent patterns.

        Returns:
            tf.Tensor: Absolute positional encoding matrix of shape (1, position, d_model)
        """
        # Create position and dimension indices
        position_idx = np.arange(self.position)[:, np.newaxis]
        dim_idx = np.arange(self.d_model)[np.newaxis, :]

        # Calculate angles
        angles = self.get_angles(position_idx, dim_idx, self.d_model)

        # Apply sine to even indices and cosine to odd indices
        even_idx = angles[:, 0::2]  # Even dimensions
        odd_idx = angles[:, 1::2]  # Odd dimensions

        # Create encodings using broadcasting
        pos_encoding = np.concatenate(
            [np.sin(even_idx), np.cos(odd_idx)],
            axis=-1
        )[np.newaxis, ...]

        return tf.cast(pos_encoding * self.scaling_factor, tf.float32)

    def create_relative_positional_encoding(self):
        """
        Creates the relative positional encoding matrix.
        Considers both positive and negative relative distances.

        Returns:
            tf.Tensor: Relative positional encoding matrix
        """
        # Create relative position indices from -max_relative_position to +max_relative_position
        positions = np.arange(-self.max_relative_position, self.max_relative_position + 1)
        pos_idx = positions[:, np.newaxis]
        dim_idx = np.arange(self.d_model)[np.newaxis, :]

        # Calculate angles for relative positions
        angles = self.get_angles(pos_idx, dim_idx, self.d_model)

        # Apply sine and cosine functions
        even_idx = angles[:, 0::2]
        odd_idx = angles[:, 1::2]

        rel_pos_encoding = np.concatenate(
            [np.sin(even_idx), np.cos(odd_idx)],
            axis=-1
        )

        return tf.cast(rel_pos_encoding * self.scaling_factor, tf.float32)

    def get_relative_attention_weights(self, seq_len):
        """
        Calculate relative attention weights for a given sequence length.

        Args:
            seq_len (int): Length of the input sequence

        Returns:
            tf.Tensor: Relative attention weights matrix
        """
        # Create a matrix of relative positions
        positions = tf.range(seq_len)[:, tf.newaxis] - tf.range(seq_len)[tf.newaxis, :]

        # Clip relative positions to max_relative_position
        positions = tf.clip_by_value(
            positions,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift positions to be non-negative for indexing
        positions = positions + self.max_relative_position

        # Get relative encodings for these positions
        return tf.gather(self.rel_pos_encoding, positions)

    def call(self, inputs):
        """
        Apply positional encoding to the input.
        Combines both absolute and relative position information.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            tf.Tensor: Tensor with positional encoding applied
        """
        seq_len = tf.shape(inputs)[1]

        # Get absolute positional encoding for the current sequence length
        abs_pos = self.abs_pos_encoding[:, :seq_len, :]

        # Combine absolute positions with inputs
        enhanced_inputs = inputs + abs_pos

        # Add relative position information
        rel_pos = self.get_relative_attention_weights(seq_len)

        # Scale the output
        output = enhanced_inputs * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        return output

    def get_config(self):
        """
        Get configuration for serialization.

        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
            'max_relative_position': self.max_relative_position,
            'scaling_factor': self.scaling_factor
        })
        return config
    @classmethod
    def from_config(cls, config):
        """
        Create a layer instance from its config.
        """
        return cls(**config)

def transformer_block(inputs, num_heads, d_model, dff, dropout_rate):
    """
    Enhanced transformer block with multi-head attention and additional features
    """
    # Multi-head attention with dropout
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model//num_heads,  # Scaled key dimension
        dropout=dropout_rate
    )(inputs, inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Enhanced feed-forward network
    ffn_output = point_wise_feed_forward_network(out1, dff, d_model, dropout_rate)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

def point_wise_feed_forward_network(x, dff, d_model, dropout_rate):
    """
    Enhanced feed-forward network with additional activation and dropout
    """
    x = Dense(dff, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dff//2, activation='gelu')(x)  # Additional layer with GELU
    x = Dropout(dropout_rate)(x)
    return Dense(d_model)(x)

@register_keras_serializable()
def positional_encoding(sequence_length, d_model):
    """
    Create positional encodings for the transformer.

    Args:
        sequence_length: Length of the input sequence
        d_model: Dimension of the model embeddings

    Returns:
        Tensor with shape (1, sequence_length, d_model) containing positional encodings
    """
    # Create position indices
    positions = tf.range(start=0, limit=sequence_length, delta=1, dtype=tf.float32)
    positions = positions[:, tf.newaxis]  # Shape: (sequence_length, 1)

    # Create dimension indices
    i = tf.range(start=0, limit=d_model, delta=2, dtype=tf.float32)
    i = i[tf.newaxis, :]  # Shape: (1, d_model/2)

    # Calculate angle rates
    angle_rates = 1 / tf.pow(10000.0, (i / tf.cast(d_model, tf.float32)))

    # Calculate angles
    angle_rads = positions * angle_rates  # Shape: (sequence_length, d_model/2)

    # Apply sin and cos
    sin_values = tf.sin(angle_rads)  # (sequence_length, d_model/2)
    cos_values = tf.cos(angle_rads)  # (sequence_length, d_model/2)

    # Interleave sin and cos values
    pos_encoding = tf.stack([sin_values, cos_values], axis=2)  # (sequence_length, d_model/2, 2)
    pos_encoding = tf.reshape(pos_encoding, (sequence_length, d_model))  # (sequence_length, d_model)

    # Add batch dimension
    pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, sequence_length, d_model)

    return pos_encoding


def get_angles(positions, i, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return positions * angle_rates


@register_keras_serializable()
class GPKernelAttention(Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(GPKernelAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Initialize these as None - they will be created in build()
        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=1.0)

    def build(self, input_shape):
        # Create the layers
        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)
        self.dense = Dense(self.d_model)

        # Build each dense layer
        input_shape_dense = tf.TensorShape([input_shape[0], input_shape[1], input_shape[2]])
        self.wq.build(input_shape_dense)
        self.wk.build(input_shape_dense)
        self.wv.build(input_shape_dense)
        self.dense.build(tf.TensorShape([input_shape[0], input_shape[1], self.d_model]))

        # Initialize the trainable weights list using the parent class's mechanism
        self._trainable_weights = []

        # Add weights from each dense layer
        self._trainable_weights.extend(self.wq.trainable_weights)
        self._trainable_weights.extend(self.wk.trainable_weights)
        self._trainable_weights.extend(self.wv.trainable_weights)
        self._trainable_weights.extend(self.dense.trainable_weights)

        super(GPKernelAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, mask=None):
        # Unpack inputs if they're passed as a list/tuple
        if isinstance(inputs, (list, tuple)):
            q, k, v = inputs
        else:
            q = k = v = inputs

        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Reshape for kernel computation
        q = tf.reshape(q, (batch_size * self.num_heads, -1, self.depth))
        k = tf.reshape(k, (batch_size * self.num_heads, -1, self.depth))
        v = tf.reshape(v, (batch_size * self.num_heads, -1, self.depth))

        # Compute pairwise distances
        dists = (
                tf.reduce_sum(q ** 2, axis=2, keepdims=True)
                - 2 * tf.matmul(q, k, transpose_b=True)
                + tf.transpose(tf.reduce_sum(k ** 2, axis=2, keepdims=True), [0, 2, 1])
        )

        # RBF kernel
        attn = tf.exp(-0.5 * dists / (self.kernel.length_scale ** 2))

        # Optional mask
        if mask is not None:
            attn += (mask * -1e9)

        # Softmax
        attn = tf.nn.softmax(attn, axis=-1)

        # Weighted sum
        output = tf.matmul(attn, v)

        # Reshape back
        output = tf.reshape(
            output,
            (batch_size, self.num_heads, -1, self.depth)
        )
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)

    def get_config(self):
        config = super(GPKernelAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
def gp_transformer_block(inputs, d_model, num_heads, dff):
    """
    A single transformer block using Gaussian Process Kernel Attention.
    """
    # GP Attention
    attention_layer = GPKernelAttention(d_model, num_heads)
    gp_attn = attention_layer(inputs)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + gp_attn)

    # Feed Forward
    ffn = Dense(dff, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)

    return out2

###############################################################################
# 3) TRANSFORMER VARIANTS
###############################################################################

def transformer_positional(
    inp_shape,
    tgt_shape,
    model_name='transformer_positional',
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.1,
):
    inputs = Input(shape=(inp_shape[0], 1))

    # Embedding layer
    x = Conv1D(d_model, 1, activation="linear")(inputs)

    # Positional encoding
    sequence_length = inp_shape[0]
    pos_encoding = positional_encoding(sequence_length, d_model)
    x = x + pos_encoding

    # Transformer blocks
    for _ in range(4):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)(x, x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed Forward Network
        ffn_output = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])(out1)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Sequence to sequence decoder
    decoder = GRU(d_model, return_sequences=True)(x)

    # Final output layer - select only the last tgt_shape[0] timesteps
    outputs = Conv1D(filters=1, kernel_size=1)(decoder[:, -tgt_shape[0]:, :])

    model = Model(inputs, outputs,name=model_name)
    return model


def transformer_gaussian(
    inp_shape,
    tgt_shape,
    num_heads=4,
    d_model=64,
    dff=128,
    model_name='transformer_gaussian',
    n_layers=2
):
    """
    Transformer with Gaussian Kernel Attention (no positional encoding).
    """
    inputs = Input(shape=(inp_shape[0], 1))

    # Project input to d_model
    x = Dense(d_model)(inputs)

    # Stacking GP-based transformer blocks
    for _ in range(n_layers):
        x = gp_transformer_block(x, d_model, num_heads, dff)

    # Output
    x = Flatten()(x)
    outputs = Dense(tgt_shape[-1])(x)

    model = Model(inputs, outputs,name=model_name)
    return model


def transformer_positional_gaussian(
    inp_shape,
    tgt_shape,
    num_heads=4,
    d_model=64,
    dff=128,
    model_name = 'transformer_positional_gaussian',
    n_layers=2

):
    """
    Transformer with Gaussian Kernel Attention + Positional Encoding.
    """
    inputs = Input(shape=(inp_shape[0], 1))

    # Project input to d_model
    x = Dense(d_model)(inputs)

    # Positional Encoding
    x = PositionalEncoding(inp_shape[0], d_model)(x)

    # Stacking GP-based transformer blocks
    for _ in range(n_layers):
        x = gp_transformer_block(x, d_model, num_heads, dff)

    # Output
    x = Flatten()(x)
    outputs = Dense(tgt_shape[-1])(x)

    model = Model(inputs, outputs,name=model_name)
    return model
