from pathlib import Path

from polycraft_nov_data.dataloader import polycraft_dataloaders

import polycraft_nov_det.mnist_loader as mnist_loader
import polycraft_nov_det.model_utils as model_utils
import polycraft_nov_det.novelty as novelty
import polycraft_nov_det.plot as plot


def eval_mnist(model_path, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_mnist_model(model_path, device).eval()
    dataloaders = mnist_loader.torch_mnist(include_novel=True)
    eval(model_path, model, dataloaders, device)


def eval_polycraft(model_path, device="cpu"):
    model_path = Path(model_path)
    model = model_utils.load_polycraft_model(model_path, device).eval()
    dataloaders = polycraft_dataloaders(include_novel=True)
    eval(model_path, model, dataloaders, device)


def eval(model_path, model, dataloaders, device="cpu"):
    train_loader, valid_loader, test_loader = dataloaders
    eval_path = Path(model_path.parent, "eval_" + model_path.stem)
    eval_path.mkdir(exist_ok=True)
    # construct and plot ECDF
    train_ecdf = model_utils.load_cached_ecdf(model_path, model, train_loader)
    plot.plot_empirical_cdf(train_ecdf).savefig(eval_path / Path("ecdf.png"))
    # fit detector threshold on validation set
    detector = novelty.ReconstructionDet(model, train_ecdf, device)
