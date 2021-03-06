import matplotlib.pyplot as plt

import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.novelty as novelty
import polycraft_nov_det.plot as plot


def eval_mnist(model_path):
    model = model_utils.load_mnist_model(model_path)
    train_ecdf = model_utils.load_cached_ecdf(model_path, model)
    plot.plot_empirical_cdf(train_ecdf)
    detector = novelty.ReconstructionDet(model, train_ecdf)
    train_loader, _, _ = mnist_loader.torch_mnist(6)
    data, _ = next(iter(train_loader))
    plot.plot_per_patch_nov_det(detector, .99, (2, 3, 1, 28, 28), data)
    plt.show()
