import pytest
import numpy as np
import tensorflow as tf
from variart.preprocessing import ArtVideo
from variart.model import VAE
from variart.latent import Latent

# Data
data = np.random.rand(16, 16, 16).astype("float32")

# Model
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
model = VAE("VAE", latent_dim, input_shape, inference_net, generative_net)

def test_name():

    # Given
    name = "LatentObject"

    # When
    LatentObject = Latent(data, model, name=name)

    # Then
    assert LatentObject.name == name

def test_encode():
    # Given
    LatentObject = Latent(data, model)

    # When
    LatentObject.encode_data()

    # Then
    assert LatentObject.Z.shape == (16, latent_dim)

def test_decode():
    # Given
    LatentObject = Latent(data, model)

    # When
    LatentObject.encode_data()
    LatentObject.decode_data()

    # Then
    assert LatentObject.Z_decoded.shape == data.shape

def test_tsne():
    # Given
    LatentObject = Latent(data, model)

    # When
    LatentObject.encode_data()
    LatentObject.latent_tsne()

    # Then
    assert LatentObject.Z_tsne.shape == (16, 2)