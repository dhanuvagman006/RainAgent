import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Dense, 
    Flatten, Input, Dropout, GlobalAveragePooling1D, 
    LayerNormalization, MultiHeadAttention, BatchNormalization
)

def build_lstm(input_shape, y_days):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_days, activation='relu')
    ], name='LSTM')
    return model

def build_gru(input_shape, y_days):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        GRU(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_days, activation='relu')
    ], name='GRU')
    return model

def build_bilstm(input_shape, y_days):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_days, activation='relu')
    ], name='Bi-LSTM')
    return model

def build_1d_cnn(input_shape, y_days):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y_days, activation='relu')
    ], name='1D-CNN')
    return model

def build_cnn_lstm(input_shape, y_days):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y_days, activation='relu')
    ], name='CNN-LSTM')
    return model

def build_transformer(input_shape, y_days):
    # Time-Series Transformer block
    inputs = Input(shape=input_shape)
    
    # Self-Attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[1])(inputs, inputs)
    attention_output = Dropout(0.2)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward
    ffn_output = Dense(128, activation='relu')(out1)
    ffn_output = Dense(input_shape[1])(ffn_output)
    ffn_output = Dropout(0.2)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    # Global Average Pooling and Output
    x = GlobalAveragePooling1D()(out2)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(y_days, activation='relu')(x)
    
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
