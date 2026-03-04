from __future__ import annotations

import tensorflow as tf


def reconstruction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    smooth = tf.reduce_mean(tf.square(dy_true - dy_pred))
    return mse + 0.1 * smooth


def build_denoising_autoencoder(window_len: int, in_channels: int = 7) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(window_len, in_channels), name="sensor_window")
    x = tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu", name="enc_conv1")(inp)
    x = tf.keras.layers.Conv1D(4, 5, padding="same", activation="relu", name="enc_conv2")(x)
    x = tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu", name="dec_conv1")(x)
    out = tf.keras.layers.Conv1D(1, 5, padding="same", activation="linear", name="dec_conv2")(x)
    model = tf.keras.Model(inp, out, name="ppg_self_supervised_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=reconstruction_loss, metrics=["mse"])
    return model

