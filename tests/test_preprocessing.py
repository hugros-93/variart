import pytest
import numpy as np
from variart.preprocessing import ArtVideo

data = (255*np.random.rand(16, 16, 16)).astype("uint8")

def test_name():

    # Given
    name="test"
    filename=""

    # When
    Video = ArtVideo(name, filename)

    # Then
    assert Video.name == name

def test_resize():

    # Given
    name="test"
    filename=""
    Video = ArtVideo(name, filename)
    Video.X = data

    # When
    Video.resize((8,8))

    # Then
    assert Video.X.shape == Video.shape == (16,8,8)

def test_rescale():

    # Given
    name="test"
    filename=""
    Video = ArtVideo(name, filename)
    Video.X = data

    # When
    Video.rescale_image()

    # Then
    assert Video.X.max() <= 1 and Video.X.min() >= 0