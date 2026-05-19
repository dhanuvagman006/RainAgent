import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Dense,
    Flatten, Input, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, BatchNormalization,
    Add, Activation, SpatialDropout1D, AveragePooling1D,
    Concatenate, Reshape, Lambda, Multiply
)
from tensorflow.keras.regularizers import l2
import numpy as np


# ─────────────────────────────────────────────
# Helper: Residual LSTM Block
# ─────────────────────────────────────────────
def _residual_lstm_block(x, units, dropout_rate=0.2):
    """LSTM block with residual skip connection (projects if needed)."""
    lstm_out = LSTM(units, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # Project input to same size if needed for residual
    if x.shape[-1] != units:
        x = Dense(units)(x)
    out = Add()([x, lstm_out])
    return out


# ─────────────────────────────────────────────
# 1. Deep LSTM with Residual Connections
# ─────────────────────────────────────────────
def build_lstm(input_shape, y_days):
    """Deep LSTM with residual connections + soft attention — NSE-optimised."""
    inputs = Input(shape=input_shape)

    # Stem projection
    x = Dense(256, kernel_regularizer=l2(1e-5))(inputs)

    # Block 1
    h1 = LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    h1 = BatchNormalization()(h1)
    h1 = Dropout(0.10)(h1)
    x  = Add()([x, h1])   # residual

    # Block 2
    h2 = LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    h2 = BatchNormalization()(h2)
    h2 = Dropout(0.10)(h2)
    x  = Add()([x, h2])   # residual

    # Block 3 — compress
    x  = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    x  = BatchNormalization()(x)
    x  = Dropout(0.10)(x)

    # Soft attention pooling over time axis
    attn = Dense(1, activation='softmax')(x)   # (batch, T, 1)
    x    = Multiply()([x, attn])
    x    = Lambda(lambda t: keras.ops.sum(t, axis=1))(x)  # (batch, 128)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.10)(x)
    x = Dense(64,  activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    return model


# ─────────────────────────────────────────────
# 2. Deep Stacked GRU
# ─────────────────────────────────────────────
def build_gru(input_shape, y_days):
    """Deep GRU with residual connections + soft attention — NSE-optimised."""
    inputs = Input(shape=input_shape)

    # Stem projection
    x = Dense(256, kernel_regularizer=l2(1e-5))(inputs)

    # Block 1
    h1 = GRU(256, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    h1 = BatchNormalization()(h1)
    h1 = Dropout(0.10)(h1)
    x  = Add()([x, h1])

    # Block 2
    h2 = GRU(256, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    h2 = BatchNormalization()(h2)
    h2 = Dropout(0.10)(h2)
    x  = Add()([x, h2])

    # Block 3 — compress
    x  = GRU(128, return_sequences=True, kernel_regularizer=l2(1e-5))(x)
    x  = BatchNormalization()(x)
    x  = Dropout(0.10)(x)

    # Soft attention pooling
    attn = Dense(1, activation='softmax')(x)
    x    = Multiply()([x, attn])
    x    = Lambda(lambda t: keras.ops.sum(t, axis=1))(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.10)(x)
    x = Dense(64,  activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='GRU')
    return model


# ─────────────────────────────────────────────
# 3. Bidirectional LSTM with Attention
# ─────────────────────────────────────────────
def build_bilstm(input_shape, y_days):
    """Bidirectional LSTM with increased capacity and soft attention."""
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(1e-5)))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.10)(x)

    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(1e-5)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.10)(x)

    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(1e-5)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.10)(x)

    # Soft attention pooling
    attn_weights = Dense(1, activation='softmax')(x)
    x = Multiply()([x, attn_weights])
    x = Lambda(lambda t: keras.ops.sum(t, axis=1))(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.10)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Bi-LSTM')
    return model


# ─────────────────────────────────────────────
# 4. Deep Dilated Temporal CNN (TCN-style)
# ─────────────────────────────────────────────
def build_1d_cnn(input_shape, y_days):
    inputs = Input(shape=input_shape)

    # Multi-scale temporal feature extraction
    def dilated_block(x, filters, dilation):
        conv = Conv1D(filters, kernel_size=3, dilation_rate=dilation,
                      padding='causal', activation='relu',
                      kernel_regularizer=l2(1e-4))(x)
        conv = BatchNormalization()(conv)
        # Residual projection if channel size differs
        if x.shape[-1] != filters:
            x = Conv1D(filters, kernel_size=1, padding='same')(x)
        return Add()([x, conv])

    x = Conv1D(64, kernel_size=1, padding='causal', activation='relu')(inputs)  # stem

    x = dilated_block(x, 64, dilation=1)
    x = Dropout(0.15)(x)
    x = dilated_block(x, 64, dilation=2)
    x = Dropout(0.15)(x)
    x = dilated_block(x, 128, dilation=4)
    x = Dropout(0.15)(x)
    x = dilated_block(x, 128, dilation=8)
    x = Dropout(0.15)(x)
    x = dilated_block(x, 128, dilation=16)
    x = Dropout(0.15)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='1D-CNN')
    return model


# ─────────────────────────────────────────────
# 5. CNN-LSTM Hybrid with Attention
# ─────────────────────────────────────────────
def build_cnn_lstm(input_shape, y_days):
    inputs = Input(shape=input_shape)

    # CNN feature extractor
    x = Conv1D(64, kernel_size=3, activation='relu', padding='causal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='causal',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, kernel_size=1, activation='relu', padding='causal')(x)
    x = BatchNormalization()(x)

    # LSTM temporal modelling
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Attention pooling (pure Keras ops)
    attn_weights = Dense(1, activation='softmax')(x)
    x = Multiply()([x, attn_weights])
    x = Lambda(lambda t: keras.ops.sum(t, axis=1))(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN-LSTM')
    return model


# ─────────────────────────────────────────────
# 6. Multi-Head Transformer with Positional Encoding
# ─────────────────────────────────────────────
def _positional_encoding(seq_len, d_model):
    """Generates a fixed sinusoidal positional encoding tensor."""
    positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
    dims = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
    angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_enc = tf.concat([sines, cosines], axis=-1)
    return pos_enc[tf.newaxis, :, :]  # (1, seq_len, d_model)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_enc = _positional_encoding(seq_len, d_model)

    def call(self, x):
        return x + tf.cast(self.pos_enc[:, :tf.shape(x)[1], :], x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return config


def _transformer_block(x, d_model, num_heads, ff_dim, dropout_rate=0.1):
    """Single Transformer encoder block."""
    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                  dropout=dropout_rate)(x, x)
    attn_out = Dropout(dropout_rate)(attn_out)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_out)

    # Point-wise FFN
    ffn = Dense(ff_dim, activation='relu', kernel_regularizer=l2(1e-4))(out1)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model, kernel_regularizer=l2(1e-4))(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2


def build_transformer(input_shape, y_days):
    seq_len, n_features = input_shape
    d_model = 64

    inputs = Input(shape=input_shape)

    # Project features to d_model
    x = Dense(d_model, activation='relu', kernel_regularizer=l2(1e-4))(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)
    x = Dropout(0.1)(x)

    # 4 stacked transformer blocks
    x = _transformer_block(x, d_model=d_model, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = _transformer_block(x, d_model=d_model, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = _transformer_block(x, d_model=d_model, num_heads=4, ff_dim=128, dropout_rate=0.1)
    x = _transformer_block(x, d_model=d_model, num_heads=4, ff_dim=128, dropout_rate=0.1)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y_days, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Transformer')
    return model


def get_all_models(input_shape, y_days):
    return [
        build_lstm(input_shape, y_days),
        build_gru(input_shape, y_days),
        build_bilstm(input_shape, y_days),
        build_1d_cnn(input_shape, y_days),
        build_cnn_lstm(input_shape, y_days),
        build_transformer(input_shape, y_days)
    ]
