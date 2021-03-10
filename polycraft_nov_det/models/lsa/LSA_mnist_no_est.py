from importlib.resources import path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_det.data import torch_mnist
import polycraft_nov_det.models.lsa.unmodified.models.LSA_mnist as LSA_mnist
import polycraft_nov_det.models.lsa.unmodified.models.base as base
from polycraft_nov_det.plot import plot_reconstruction


class LSAMNISTNoEst(base.BaseModule):
    """
    LSA model for MNIST one-class classification without estimator.
    """
    def __init__(self,  input_shape, code_length):
        """Class constructor.

        Args:
            input_shape (Tuple[int, int, int]): the shape of MNIST samples.
            code_length (int): the dimensionality of latent vectors.
        """
        super(LSAMNISTNoEst, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        # Build encoder
        self.encoder = LSA_mnist.Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = LSA_mnist.Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

    def forward(self, x):
        """Forward propagation.

        Args:
            x (torch.Tensor): the input batch of images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of torch.Tensors holding reconstructions,
                                               and latent vectors.
        """
        h = x

        # Produce representations
        z = self.encoder(h)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z


def train():
    # define shape constants
    mnist_input_shape = (1, 28, 28)
    batch_size = 256
    # get dataloaders
    train_loader, valid_loader, _ = torch_mnist(batch_size)
    # get Tensorboard writer
    with path("polycraft_nov_det.models.lsa", "runs") as log_dir:
        writer = SummaryWriter(log_dir)
    # define training constants
    lr = 1e-3
    epochs = 100
    loss_func = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct model
    model = LSAMNISTNoEst(mnist_input_shape, 64)
    model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # train model
    for epoch in range(epochs):
        train_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # update weights with optimizer
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            optimizer.step()
            # logging
            train_loss += batch_loss.item() * batch_size
        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        # get validation loss
        valid_loss = 0
        for data, target in valid_loader:
            data = data.to(device)
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            valid_loss += batch_loss.item() * batch_size
        av_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)
        # get reconstruction visualization
        writer.add_figure("Reconstruction Vis", plot_reconstruction(data, r_data))
        # TODO add latent space visualization (try PCA or t-SNE for projection)
        # save model
        if (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), "LSA_mnist_no_est_%d.pt" % (epoch + 1,))
    return model
