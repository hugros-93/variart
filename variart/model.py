import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from IPython import display
import plotly.express as px
from plotly.subplots import make_subplots

from .preprocessing import rescale_image

# References: 
# - https://www.tensorflow.org/tutorials/generative/cvae
# - https://www.tensorflow.org/tutorials/generative/dcgan

# VAE #

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def apply_gradients(optimizer, gradients, variables):
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
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent)
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
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
            new_img = rescale_image(data_validation[i])
            fig.add_trace(
                px.imshow(new_img).data[0],
                row=1,
                col=i + 1,
            )
            new_img = rescale_image(x_logit[i])
            fig.add_trace(
                px.imshow(new_img).data[0],
                row=2,
                col=i + 1,
            )
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

    def load_model(self, batch_size):
        # load json and create model
        json_file = open(f"{self.name_model}_encoder.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.inference_net = tf.keras.models.model_from_json(loaded_model_json)
        self.inference_net.build(input_shape=self.input_shape_tuple)

        json_file = open(f"{self.name_model}_decoder.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.generative_net = tf.keras.models.model_from_json(loaded_model_json)
        self.generative_net.build(input_shape=(batch_size, self.latent_dim))

        # load weights into new model
        self.inference_net.load_weights(f"{self.name_model}_encoder.h5")
        self.generative_net.load_weights(f"{self.name_model}_decoder.h5")
        print("Loaded model from disk")

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

        :param optimizer: tensorflow optimizer
        :param train_dataset: tensorflow dataset for training
        :param validation_dataset: tensorflow dataset for validation
        :param epochs: number of epochs
        :param batch_size: number of element for each batch
        :param early_stop_patience: number of iteration without improvement on the validation set before stopping
        :param freq_plot: frequency for image plot
        :param plot_test: plot test images if True
        :param n_to_plot: number of val. image to plot (encoded/decoded)
        :return: trained model

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
                data_validation = np.concatenate(
                    [np.array(x) for x in validation_dataset]
                )
                x_logit = self.decode(
                    np.concatenate(test_vector_for_generation), apply_sigmoid=True
                )
                self.plot_training_images(
                    data_validation,
                    x_logit,
                    self.n_to_plot,
                )
        return self

# GAN #

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output, wgan=False):
    if wgan:
        return tf.reduce_mean(1-fake_output)
    else:
        return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output, wgan=False):
    if wgan:
        real_loss = tf.reduce_mean(1-real_output)
        fake_loss = tf.reduce_mean(fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

class GAN(tf.keras.Model):
    """
    Class to define a Generative Adversial Network (GAN)
    """
    def __init__(self, name_model, noise_dim, input_shape_tuple, generator, discriminator, 
    learning_rate = 1e-4, wgan=False):
        super().__init__()
        self.name_model = name_model
        self.noise_dim = noise_dim
        self.input_shape_tuple = input_shape_tuple
        self.generator = generator
        self.discriminator = discriminator
        self.wgan = wgan
        self.learning_rate = learning_rate

        if wgan:
            self.generator_optimizer = RMSprop(learning_rate=learning_rate)
            self.discriminator_optimizer = RMSprop(learning_rate=learning_rate, clipvalue=0.01)
        else:
            self.generator_optimizer = Adam(learning_rate)
            self.discriminator_optimizer = Adam(learning_rate)

    @tf.function
    def generator_train_step(self, images):
        noise = tf.random.normal([images.shape[0], self.noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output, wgan=self.wgan)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    @tf.function
    def discriminator_train_step(self, images):
        noise = tf.random.normal([images.shape[0], self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output, wgan=self.wgan)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss

    def print_epoch(self, epoch):
        print (f'Epoch {epoch+1} \t|\t gen_loss={self.gen_loss} \t|\t disc_loss={self.disc_loss}')

    def generate_and_plot(self, n_to_plot, return_fig=False):
        fig = make_subplots(rows=1, cols=n_to_plot)
        for i in range(n_to_plot):
            noise = tf.random.normal([1, self.noise_dim])
            generated_image = self.generator(noise, training=False)[0]
            new_img = rescale_image(generated_image)
            fig.add_trace(
                px.imshow(new_img).data[0],
                row=1,
                col=i + 1,
            )
        fig.update_layout(coloraxis_showscale=False, hovermode=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        display.clear_output(wait=True)
        fig.show()
        if return_fig:
            return fig

    def _save_network(self):
        # serialize model to JSON
        model_json_generator = self.generator.to_json()
        model_json_discriminator = self.discriminator.to_json()
        with open(f"{self.name_model}_generator.json", "w") as json_file:
            json_file.write(model_json_generator)
        with open(f"{self.name_model}_discriminator.json", "w") as json_file:
            json_file.write(model_json_discriminator)
        # serialize weights to HDF5
        self.generator.save_weights(f"{self.name_model}_generator.h5")
        self.discriminator.save_weights(f"{self.name_model}_discriminator.h5")

    def load_model(self, batch_size):
        # load json and create model
        json_file = open(f"{self.name_model}_generator.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.generator = tf.keras.models.model_from_json(loaded_model_json)
        self.generator.build(input_shape=self.noise_dim)

        json_file = open(f"{self.name_model}_discriminator.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.discriminator = tf.keras.models.model_from_json(loaded_model_json)
        self.discriminator.build(input_shape=(batch_size, self.input_shape_tuple))

        # load weights into new model
        self.generator.load_weights(f"{self.name_model}_generator.h5")
        self.discriminator.load_weights(f"{self.name_model}_discriminator.h5")
        print("Loaded model from disk")

    def train(self, dataset, epochs, n_steps_gen=1, n_steps_disc=None, freq_plot=None, n_to_plot=4):
        if not n_steps_gen and not n_steps_disc:
            for epoch in range(epochs):
                for image_batch in dataset:
                    self.disc_loss = self.discriminator_train_step(image_batch)
                    self.gen_loss = self.generator_train_step(image_batch)
                self.print_epoch(epoch)
                self._save_network()
                if (epoch+1) % freq_plot == 0:
                    self.generate_and_plot(n_to_plot)
        else:
            if not n_steps_gen:
                n_steps_gen=1
            if not n_steps_disc:
                n_steps_disc=1
            for epoch in range(epochs):
                for n in range(n_steps_disc):
                    for image_batch in dataset:
                        self.disc_loss = self.discriminator_train_step(image_batch)
                for n in range(n_steps_gen):
                    for image_batch in dataset:
                        self.gen_loss = self.generator_train_step(image_batch)
                self.print_epoch(epoch)
                self._save_network()
                if (epoch+1) % freq_plot == 0:
                    self.generate_and_plot(n_to_plot)