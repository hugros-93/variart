import os
import cv2
import numpy as np
import random
import plotly.express as px
import matplotlib.pyplot as plt

class ArtObject:
    """
    Class to define a general art objects.
    """

    def __init__(self):
        self.type = None
        self.name = None
        self.X = None
        self.shape = []

    def rescale_image(self):
        self.X = np.interp(self.X, (0, 255), (0, 1))

    def show_random_image(self):
        i = random.randint(0, self.shape[0] - 1)
        img = self.X[i]
        fig = px.imshow(np.interp(img, (0, 1), (0, 255)))
        fig.update_layout(
            coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()

    def square(self):
        limit = abs(int((self.shape[2] - self.shape[1]) / 2))
        if self.shape[1] < self.shape[2]:
            self.X = self.X[:, :, limit : limit + self.shape[1], :]
        elif self.shape[1] > self.shape[2]:
            self.X = self.X[:, limit : limit + self.shape[2], :, :]

        self.shape = self.X.shape

    def resize(self, new_shape):
        list_data = []
        for x in self.X:
            new_x = cv2.resize(
                x, dsize=(new_shape[0], new_shape[1]), interpolation=cv2.INTER_CUBIC
            )
            list_data.append(new_x)
        new_X = np.array(list_data)

        self.X = new_X
        self.shape = new_X.shape

class ArtPictures(ArtObject):
    """
    Class to define picture object.
    """

    def __init__(self, name, filename):
        self.type = "pictures"
        self.name = name
        self.filename = filename

    def load_pictures(self):
        list_files = os.listdir(self.filename)
        list_data = []
        for file in list_files:
            data = plt.imread(f'{self.filename}{file}')
            list_data.append(data)
        self.X = np.swapaxes(np.array(list_data),1,2)
        self.shape = self.X.shape

class ArtVideo(ArtObject):
    """
    Class to define an art video object.
    """

    def __init__(self, name, filename):
        self.type = "video"
        self.name = name
        self.filename = filename

    def load_video(self):
        cap = cv2.VideoCapture(self.filename)

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        X = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

        fc = 0
        ret = True

        while fc < frameCount and ret:
            ret, X[fc] = cap.read()
            fc += 1

        cap.release()

        self.X = X
        self.shape = X.shape