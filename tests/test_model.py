import pytest
import numpy as np
import tensorflow as tf
from variart.model import VAE

data = np.random.rand(16, 16, 16).astype("float32")
batch_size = 4
latent_dim = 2
input_shape = (batch_size, data.shape[0], data.shape[1])
inference_net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2*latent_dim)
])
generative_net = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape[1]*input_shape[2]),
    tf.keras.layers.Reshape((input_shape[1],input_shape[2]))
])

def test_name():

    # Given
    name="VAE"

    # When
    model = VAE(name, latent_dim, input_shape, inference_net, generative_net)

    # Then
    assert model.name_model == name

def test_train_shape_latent():

    # Given
    model = VAE("VAE", latent_dim, input_shape, inference_net, generative_net)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    epochs = 1

    # When
    model = model.train(
        optimizer,
        train_dataset,
        validation_dataset,
        epochs,
        batch_size,
        freq_plot=epochs + 1,
    )
    mean, std = model.encode(data)

    # Then
    assert mean.shape[1] == std.shape[1] == latent_dim