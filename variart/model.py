import json
import math
import numpy as np
import tensorflow as tf
from IPython import display
import plotly.express as px
from plotly.subplots import make_subplots


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Compute log normal density
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def apply_gradients(optimizer, gradients, variables):
    """
    Apply gradient to variables
    """
    optimizer.apply_gradients(zip(gradients, variables))


class VAE(tf.keras.Model):
    """
    Class to define a Variational AutoEncoder (VAE).
    """

    def __init__(
        self, name_model, latent_dim, input_shape_tuple, inference_net, generative_net
    ):
        super().__init__()
        self.name_model = name_model
        self.latent_dim = latent_dim
        self.input_shape_tuple = input_shape_tuple
        self.inference_net = inference_net
        self.generative_net = generative_net

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        # x_logit = tf.cast(x_logit, tf.float32)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

        logpx_z = -tf.reduce_sum(cross_ent)
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)

        # logpx_z = tf.cast(logpx_z, tf.float32)
        # logqz_x = tf.cast(logqz_x, tf.float32)
        # logpz = tf.cast(logpz, tf.float32)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def sample(self, eps):
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def plot_training_images(self, data_validation, x_logit, n_to_plot):
        fig = make_subplots(rows=2, cols=n_to_plot)
        for i in range(n_to_plot):
            fig.add_trace(px.imshow(data_validation[i]).data[0], row=1, col=i + 1)
            fig.add_trace(px.imshow(x_logit[i]).data[0], row=2, col=i + 1)
        fig.update_layout(coloraxis_showscale=False, hovermode=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        display.clear_output(wait=True)
        fig.show()

    def _save_network(self):
        # serialize model to JSON
        model_json_encoder = self.inference_net.to_json()
        model_json_decoder = self.generative_net.to_json()
        with open(f"{self.name_model}_encoder.json", "w") as json_file:
            json_file.write(model_json_encoder)
        with open(f"{self.name_model}_decoder.json", "w") as json_file:
            json_file.write(model_json_decoder)
        # serialize weights to HDF5
        self.inference_net.save_weights(f"{self.name_model}_encoder.h5")
        self.generative_net.save_weights(f"{self.name_model}_decoder.h5")

    def _early_stop(self):

        print(f"early stop : {self.epoch}")

        # load json and create model
        with open(f"{self.name_model}_encoder.json", "r") as f:
            loaded_model_json = f.read()
        self.inference_net = tf.keras.models.model_from_json(loaded_model_json)
        self.inference_net.build(input_shape=self.input_shape_tuple)

        with open(f"{self.name_model}_decoder.json", "r") as f:
            loaded_model_json = f.read()
        self.generative_net = tf.keras.models.model_from_json(loaded_model_json)
        self.generative_net.build(input_shape=(self.batch_size, self.latent_dim))

        # load weights into new model
        self.inference_net.load_weights(f"{self.name_model}_encoder.h5")
        self.generative_net.load_weights(f"{self.name_model}_decoder.h5")
        print("Loaded model from disk")

    def train(
        self,
        optimizer,
        train_dataset,
        validation_dataset,
        epochs,
        batch_size,
        early_stop_patience=50,
        freq_plot=10,
        plot_test=False,
        n_to_plot=1,
    ):
        """
        Train VAE

        :param optimizer: tf optimizer
        :param train_dataset: tf dataset
        :param validation_dataset: tf dataset
        :param epochs: number of epochs
        :param batch_size: number of element for each batch
        :param early_stop_patience: number of iteration without improvement in validation set before stopping
        :param freq_plot: frequency for image plot
        :param plot_test: plot test images if True
        :param n_to_plot: number of val. image to plot
        :return: trained VAE

        """

        self.freq_plot = freq_plot
        self.batch_size = batch_size
        self.plot_test = plot_test
        self.elbo = math.inf
        self.train_elbo = math.inf
        self.nb_features = self.input_shape_tuple[1]
        self.epoch = 0
        self.num_examples_to_generate = 20
        self.elbo_list = []
        self.train_elbo_list = []
        self.n_to_plot = n_to_plot

        early_stop_test = 0
        loss_before = 1e10

        for epoch in range(1, epochs + 1):

            self.epoch = epoch
            train_loss = tf.keras.metrics.Mean()
            test_vector_for_generation = []

            for train_x in train_dataset:
                gradients, loss = self.compute_gradients(train_x)
                apply_gradients(optimizer, gradients, self.trainable_variables)
                train_loss(self.compute_loss(train_x))

            loss = tf.keras.metrics.Mean()

            for test_x in validation_dataset:
                loss(self.compute_loss(test_x))
                mean, logvar = self.encode(test_x)
                z = self.reparameterize(mean, logvar)
                test_vector_for_generation.append(z)

            self.elbo = loss.result()
            self.elbo_list.append(self.elbo)

            self.train_elbo = train_loss.result()
            self.train_elbo_list.append(self.train_elbo)

            # Print
            print(
                f"Epoch: {self.epoch} \nTrain set Loss: {self.train_elbo}, \nValidation set Loss: {self.elbo}"
            )
            # Early stop
            if early_stop_patience > 0:
                if self.elbo > loss_before:
                    print("\t> no improvement...")
                    early_stop_test += 1
                    if early_stop_test >= early_stop_patience:
                        self._early_stop()
                        break
                else:
                    print("\t> improvement !")
                    early_stop_test = 0
                    loss_before = self.elbo
                    self._save_network()

            if epoch % self.freq_plot == 0 and epoch > 0 and plot_test:
                data_validation = np.concatenate([np.array(x) for x in validation_dataset])
                x_logit = self.decode(
                    np.concatenate(test_vector_for_generation), apply_sigmoid=True
                )
                self.plot_training_images(
                    data_validation,
                    x_logit,
                    self.n_to_plot,
                )
        return self