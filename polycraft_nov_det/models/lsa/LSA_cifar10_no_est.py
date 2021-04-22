import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.polycraft_dataloader import create_data_generators

import polycraft_nov_det.models.lsa.unmodified.models.LSA_cifar10 as LSA_cifar10
import polycraft_nov_det.models.lsa.unmodified.models.base as base
from polycraft_nov_det.plot import plot_reconstruction


class LSACIFAR10NoEst(base.BaseModule):
    """
    LSA model for CIFAR10 one-class classification without estimator.
    """

    def __init__(self,  input_shape, code_length):
        """Class constructor.

        Args:
            input_shape (Tuple[int, int, int]): the shape of CIFAR10 samples.
            code_length (int): the dimensionality of latent vectors.
        """
        super(LSACIFAR10NoEst, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        # Build encoder
        self.encoder = LSA_cifar10.Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = LSA_cifar10.Decoder(
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
        # Produce representations
        z = self.encoder(x)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z


def train():
    # batch size depends on scale we use 0.25 --> 6, 0.5 --> 42, 0.75 --> 110, 1 --> 195
    scale = 0.75
    batch_size = 110
    n_z = 100

    pc_input_shape = (3, 32, 32)  # color channels, height, width

    # get dataloaders
    train_loader, valid_loader, test_loader = create_data_generators(
        shuffle=True,
        novelty_type='normal', scale_level=scale)

    print('Size of training loader', len(train_loader), flush=True)
    print('Size of validation loader', len(valid_loader), flush=True)
    print('Size of test loader', len(test_loader), flush=True)

    # get Tensorboard writer
    writer = SummaryWriter("runs_0_75")

    # define training constants
    lr = 1e-3
    epochs = 1000
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # construct model
    model = LSACIFAR10NoEst(pc_input_shape, n_z)
    model.to(device)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # train model
    for epoch in range(epochs):

        print('---------- Epoch ', epoch, ' -------------', flush=True)

        train_loss = 0

        for i, sample in enumerate(train_loader):

            optimizer.zero_grad()

            # sample contains all patches of one screen image and its novelty description
            patches = sample[0]

            # Dimensions of flat_patches: [1, batch_size, C, H, W]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

            x = flat_patches[0].float().to(device)

            x_rec, z = model(x)

            batch_loss = loss_func(x_rec, x)
            batch_loss.backward()

            optimizer.step()

            # logging
            train_loss += batch_loss.item() * batch_size

        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        print('Average training loss  ', av_train_loss, flush=True)

        # get validation loss
        valid_loss = 0

        for i, target in enumerate(valid_loader):

            # sample contains all patches of one screen image and its novelty description
            patches = sample[0]

            # Dimensions of flat_patches: [1, batch_size, C, H, W]
            flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

            x = flat_patches[0].float().to(device)
            x_rec, z = model(x)

            batch_loss = loss_func(x, x_rec)
            valid_loss += batch_loss.item() * batch_size

        av_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)

        print('Average validation loss  ', av_valid_loss, flush=True)

        # get reconstruction visualization
        writer.add_figure("Reconstruction Vis", plot_reconstruction(x, x_rec), epoch)

        # TODO add latent space visualization (try PCA or t-SNE for projection)
        # save model
        if ((epoch + 1) % 20) == 0:
            torch.save(model.state_dict(),
                       "saved_statedict/LSA_polycraft_no_est_075_%d.pt" % (epoch + 1,))

    return model
