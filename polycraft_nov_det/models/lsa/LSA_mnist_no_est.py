import polycraft_nov_det.models.lsa.unmodified.models.LSA_mnist as LSA_mnist
import polycraft_nov_det.models.lsa.unmodified.models.base as base


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
    from importlib.resources import path

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from polycraft_nov_det.data import torch_mnist

    # define shape constants
    mnist_input_shape = (1, 28, 28)
    batch_size = 256
    # get dataloaders
    train_loader, valid_loader, _ = torch_mnist(batch_size, False)
    # get Tensorboard writer
    with path("polycraft_nov_det.models.lsa", "runs") as log_dir:
        writer = SummaryWriter(log_dir)
    # define training constants
    lr = 1e-3
    epochs = 1
    loss_func = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct model
    model = LSAMNISTNoEst(mnist_input_shape, 64)
    model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    # train model
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # update weights with optimizer
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            optimizer.step()
            # logging
            epoch_loss += batch_loss.item() * batch_size
        # record training loss
        av_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_epoch_loss, epoch)
        # TODO add reconstruction visualization
        # TODO add latent space visualization (try PCA or t-SNE for projection)
        # TODO add validation set
    # save model
    with path("polycraft_nov_det/models/lsa", "LSA_mnist_no_est.pt") as f_path:
        torch.save(model.state_dict(), f_path)
    return model
