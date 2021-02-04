import polycraft_nov_det.models.lsa.unmodified.models as models
import models.LSA_cifar10
import models.base


class LSACIFAR10NoEst(models.base.BaseModule):
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
        self.encoder = models.LSA_cifar10.Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = models.LSA_cifar10.Decoder(
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
