import matplotlib.pyplot as plt

import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.novelty as novelty
import polycraft_nov_det.plot as plot


def eval_mnist():
    model_path = "models/LSA_mnist_no_est_class_0_1_2_3_4/noisy_500_lr_1e-2/LSA_mnist_no_est_500.pt"
    model = model_utils.load_mnist_model(model_path)
    ecdfs = model_utils.load_cached_ecdfs(model_path, model)
    normal_ecdf = novelty.cat_ecdfs(ecdfs[:5])
    detector = novelty.ReconstructionDet(model, normal_ecdf)
    train_loader, _, _ = mnist_loader.torch_mnist(6, include_classes=[1])
    data, _ = next(iter(train_loader))
    plot.plot_per_patch_nov_det(detector, .99, (2, 3, 1, 28, 28), data)
    plt.show()
