import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import imageio
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm


class Latent:
    def __init__(self, data, model, name=None, scale=1.5):
        self.data = data
        self.model = model
        if not name:
            name = self.model.name_model
        self.name = name
        self.Z = None
        self.size = self.data.shape[1]
        self.scale = scale

    def encode_data(self):
        self.Z, _ = np.array(self.model.encode(self.data))
        print(f"Latent shape: {self.Z.shape}")

    def decode_data(self):
        self.Z_decoded = np.array(self.model.decode(self.Z, apply_sigmoid=True))
        print(f"Decoded shape: {self.Z_decoded.shape}")

    def latent_tsne(self, dim=2):
        tsne = TSNE(dim)
        self.Z_tsne = tsne.fit_transform(self.Z)

    def plot_latent_tsne(self, clusterer=None):
        fig = go.Figure()
        if clusterer:
            marker_color = clusterer.labels_
            text = clusterer.labels_
            txt_clust=" - after clustering"
        else:
            marker_color = "black"
            text = ""
        fig.add_trace(
            go.Scattergl(
                x=self.Z_tsne[:, 0],
                y=self.Z_tsne[:, 1],
                mode="markers",
                marker_color=marker_color,
                text=text,
            )
        )
        fig.update_layout(title=f"t-SNE projection of latent representations{txt_clust}")
        return fig

    def compute_dist_coord(self):
        self.dist_coord = np.array(
            [[z[i] for z in np.array(self.Z)] for i in range(self.Z.shape[1])]
        )

    def plot_latent_dist_coord(self):
        fig = go.Figure()
        for i, y in enumerate(self.dist_coord):
            fig.add_trace(go.Box(y=y, name=i))
        fig.update_layout(title="Distribution of latent dimensions")
        return fig

    def latent_space_clustering(self, grid):
        self.dico_clust = {}
        for n_clusters in tqdm(grid):
            clusterer = KMeans(n_clusters=n_clusters)
            clusterer.fit(self.Z)
            self.dico_clust[n_clusters] = {
                "clusterer": clusterer,
                "score": silhouette_score(self.Z, clusterer.labels_),
            }

    def plot_silhouette_score(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=[k for k in self.dico_clust],
                y=[self.dico_clust[k]["score"] for k in self.dico_clust],
                mode="markers",
                name="Clusters",
            )
        )
        fig.update_layout(
            title="Silhouette score",
            xaxis_title="nb. of clusters",
            yaxis_title="silhouette score")
        return fig

    def plot_encoded_decoded(self, list_id):
        fig = make_subplots(rows=2, cols=len(list_id))
        for i, j in enumerate(list_id):
            fig.add_trace(
                px.imshow(np.interp(self.data[j], (0, 1), (0, 255))).data[0],
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                px.imshow(np.interp(self.Z_decoded[i], (0, 1), (0, 255))).data[0],
                row=2,
                col=i + 1,
            )
        fig.update_layout(
            height=2 * self.size * self.scale,
            width=self.size * len(list_id) * self.scale,
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    def generate_image(self, n=1, c=1, method="dist", id_img=None):

        if method == "dist":
            list_z = [
                [
                    np.random.normal(np.mean(x), c * np.std(x))
                    for _, x in enumerate(self.dist_coord)
                ]
                for z in range(n)
            ]
        elif method == "random":
            list_z = [
                [np.random.normal(0, c) for _, x in enumerate(self.dist_coord)]
                for _ in range(n)
            ]
        elif method == "from_id_img" and type(id_img) == int:
            list_z = [
                self.Z[id_img] + np.random.normal(0, 1, self.Z.shape[1])
                for _ in range(n)
            ]
        else:
            raise Exception('Error in "method"')

        fig = make_subplots(rows=1, cols=n)

        for i, z in enumerate(list_z):
            decoded_img = np.array(z).reshape(1, self.model.latent_dim)
            decoded_img = self.model.decode(decoded_img, apply_sigmoid=True)[0]
            fig.add_trace(
                px.imshow(np.interp(decoded_img, (0, 1), (0, 255))).data[0],
                row=1,
                col=i + 1,
            )
        fig.update_layout(
            height=self.size * self.scale,
            width=self.size * n * self.scale,
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return list_z, fig

    def create_gif(self, list_z, filename, step=5):
        n_z = len(list_z)
        images = []
        for c in range(n_z - 1):
            z1 = list_z[c]
            z2 = list_z[c + 1]
            for i in range(step):
                z = [y + i * (x - y) / step for x, y in zip(z2, z1)]
                z = np.array(z).reshape(1, self.model.latent_dim)
                decoded_img = self.model.decode(z, apply_sigmoid=True)
                decoded_img = np.interp(np.array(decoded_img[0]), (0, 1), (0, 255))
                images.append(np.array(decoded_img, np.uint8))

        with imageio.get_writer(filename, mode="I") as writer:
            for image in images:
                writer.append_data(image)
