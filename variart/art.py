import cv2
import numpy as np
import random
import plotly.express as px

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
        i = random.randint(0, self.shape[0]-1)
        img = self.X[i]
        fig = px.imshow(img)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()


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

    def square_video(self):
        limit = abs(int((self.shape[2] - self.shape[1]) / 2))
        if self.shape[1] < self.shape[2]:
            new_X = self.X[:, :, limit : limit + self.shape[1], :]
        elif self.shape[1] > self.shape[2]:
            new_X = self.X[:, limit : limit + self.shape[2], :, :]

        self.X = new_X
        self.shape = new_X.shape

    def resize_video(self, new_shape):
        list_data = []
        for x in self.X:
            new_x = cv2.resize(
                x, dsize=(new_shape[0], new_shape[1]), interpolation=cv2.INTER_CUBIC
            )
            list_data.append(new_x)
        new_X = np.array(list_data)

        self.X = new_X
        self.shape = new_X.shape