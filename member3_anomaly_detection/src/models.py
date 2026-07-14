# models.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============ 1. LSTM AUTOENCODER ============
def create_lstm_autoencoder(sequence_length, n_features, latent_dim=8):
    """LSTM Autoencoder"""
    encoder_inputs = keras.Input(shape=(sequence_length, n_features))
    encoder = layers.LSTM(latent_dim, return_sequences=True)(encoder_inputs)
    encoder = layers.LSTM(latent_dim // 2, return_sequences=False)(encoder)
    
    decoder = layers.RepeatVector(sequence_length)(encoder)
    decoder = layers.LSTM(latent_dim // 2, return_sequences=True)(decoder)
    decoder = layers.LSTM(latent_dim, return_sequences=True)(decoder)
    decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(decoder)
    
    model = keras.Model(encoder_inputs, decoder_outputs, name='lstm_autoencoder')
    model.compile(optimizer='adam', loss='mse')
    return model

# ============ 2. VAE (Variational Autoencoder) ============
def create_vae(sequence_length, n_features, latent_dim=8):
    """VAE - learns distribution of normal behavior"""
    encoder_inputs = keras.Input(shape=(sequence_length, n_features))
    
    # Encoder
    x = layers.LSTM(latent_dim * 2, return_sequences=True)(encoder_inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    
    # Latent space
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x_dec = layers.RepeatVector(sequence_length)(decoder_inputs)
    x_dec = layers.LSTM(latent_dim, return_sequences=True)(x_dec)
    x_dec = layers.LSTM(latent_dim * 2, return_sequences=True)(x_dec)
    decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(x_dec)
    
    decoder = keras.Model(decoder_inputs, decoder_outputs)
    outputs = decoder(z)
    
    # VAE model
    vae = keras.Model(encoder_inputs, outputs, name='vae')
    
    # Custom loss
    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.mse(y_true, y_pred)
        reconstruction_loss *= sequence_length * n_features
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1)) * -0.5
        return reconstruction_loss + kl_loss
    
    vae.compile(optimizer='adam', loss=vae_loss)
    return vae


# ============ 3. TRANSFORMER AUTOENCODER ============
def create_transformer_autoencoder(sequence_length, n_features, latent_dim=8):
    """Transformer-based autoencoder"""
    encoder_inputs = keras.Input(shape=(sequence_length, n_features))
    
    # Encoder with multi-head attention
    x = layers.Dense(latent_dim)(encoder_inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=latent_dim//4)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Decoder
    decoder = layers.RepeatVector(sequence_length)(x)
    decoder = layers.Dense(latent_dim)(decoder)
    decoder = layers.MultiHeadAttention(num_heads=4, key_dim=latent_dim//4)(decoder, decoder)
    decoder = layers.LayerNormalization()(decoder)
    decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(decoder)
    
    model = keras.Model(encoder_inputs, decoder_outputs, name='transformer_autoencoder')
    model.compile(optimizer='adam', loss='mse')
    return model


# ============ 4. 1D-CNN AUTOENCODER ============
def create_cnn_autoencoder(sequence_length, n_features, latent_dim=8):
    """1D-CNN Autoencoder - lightweight"""
    encoder_inputs = keras.Input(shape=(sequence_length, n_features))
    
    # Encoder
    x = layers.Conv1D(filters=latent_dim, kernel_size=3, activation='relu', padding='same')(encoder_inputs)
    x = layers.Conv1D(filters=latent_dim//2, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Decoder
    decoder = layers.RepeatVector(sequence_length)(x)
    decoder = layers.Conv1D(filters=latent_dim//2, kernel_size=3, activation='relu', padding='same')(decoder)
    decoder = layers.Conv1D(filters=latent_dim, kernel_size=3, activation='relu', padding='same')(decoder)
    decoder_outputs = layers.Conv1D(filters=n_features, kernel_size=3, padding='same')(decoder)
    
    model = keras.Model(encoder_inputs, decoder_outputs, name='cnn_autoencoder')
    model.compile(optimizer='adam', loss='mse')
    return model