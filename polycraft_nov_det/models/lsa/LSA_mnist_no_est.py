from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_det.data import torch_mnist, GaussianNoise
import polycraft_nov_det.models.lsa.unmodified.models.LSA_mnist as LSA_mnist
import polycraft_nov_det.models.lsa.unmodified.models.base as base
from polycraft_nov_det.novelty import load_ecdf, reconstruction_ecdf
import polycraft_nov_det.plot as plot


# data shape constant
MNIST_SHAPE = (1, 28, 28)


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


def train(include_classes=None, train_noisy=True):
    """Train a model.
    Args:
        include_classes (list, optional): List of classes to include.
                                          Defaults to None, including all classes.
        train_noisy (bool, optional): Whether to use denoising autoencoder. Defaults to True.
    Returns:
        LSAMNISTNoEst: Trained model.
    """
    # get dataloaders
    batch_size = 256
    train_loader, valid_loader, _ = torch_mnist(batch_size, include_classes)
    # get Tensorboard writer
    model_label = "LSA_mnist_no_est_"
    if include_classes is None:
        model_label += "all_classes"
    else:
        classes = "_".join([str(include_class) for include_class in include_classes])
        model_label += "class_" + classes
    writer = SummaryWriter("runs/" + model_label + "/" +
                           datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    # define training constants
    lr = 1e-2
    epochs = 500
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # construct model
    model = LSAMNISTNoEst(MNIST_SHAPE, 64)
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
            if not train_noisy:
                r_data, embedding = model(data)
            else:
                r_data, embedding = model(GaussianNoise()(data))
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
        writer.add_figure("Reconstruction Vis", plot.plot_reconstruction(data, r_data, cmap="gray"),
                          epoch)
        # save model
        if (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(),
                       "models/" + model_label + "/LSA_mnist_no_est_%d.pt" % (epoch + 1,))
    return model


def load_model(path):
    """Load a saved model

    Args:
        path (str): Path to saved model state_dict

    Returns:
        LSAMNISTNoEst: Model with saved state_dict
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = LSAMNISTNoEst(MNIST_SHAPE, 64)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_cached_ecdfs(model_dir, model_name):
    model = load_model(model_dir + model_name + ".pt")
    ecdfs = []
    classes = range(10)
    for i in classes:
        ecdf_name = "%s%i_train_%s.npy" % (model_dir, i, model_name)
        try:
            ecdf = load_ecdf(ecdf_name)
        except Exception:
            train_loader, _, _ = torch_mnist(include_classes=[i])
            ecdf = reconstruction_ecdf(model, train_loader)
            ecdf.save(ecdf_name)
        ecdfs.append(ecdf)
    return ecdfs


def plot_cached_ecdfs():
    model_dir = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\"
    model_name = "LSA_mnist_no_est_500"
    return plot.plot_empirical_cdfs(load_cached_ecdfs(model_dir, model_name), range(10))


def plot_embedding():
    model_path = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\LSA_mnist_no_est_500.pt"
    model = load_model(model_path)
    embeddings = torch.Tensor([])
    targets = torch.Tensor([])
    # get embedding and targets from validation set
    _, valid_loader, _ = torch_mnist()
    for data, target in valid_loader:
        _, embedding = model(data)
        embeddings = torch.cat((embeddings, embedding))
        targets = torch.cat((targets, target))
    # plot the embedding
    plot.plot_embedding(embeddings, targets)
