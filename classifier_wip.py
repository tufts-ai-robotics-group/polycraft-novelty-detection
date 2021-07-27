import polycraft_nov_det.mnist_loader as mnist_loader
from polycraft_nov_det.models.classifier import Classifier
from polycraft_nov_det.model_utils import load_mnist_model
import polycraft_nov_det.train as train

batch_size = 64
train_loader, valid_loader, _ = mnist_loader.torch_mnist(batch_size)
# get model instances
latent_len = 64
autoencoder = load_mnist_model("models/mnist/class_0-4/noisy_500_lr_1e-2/500.pt",
                               latent_len=latent_len)
encoder = autoencoder.encoder
model = Classifier(latent_len, 5)
train.train_classifier(
    model,
    train.model_label(model),
    encoder,
    train_loader,
    valid_loader,
    .01,
    gpu=1
)
