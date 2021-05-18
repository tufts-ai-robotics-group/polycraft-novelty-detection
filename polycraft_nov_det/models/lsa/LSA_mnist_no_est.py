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
