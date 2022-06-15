import argparse

from polycraft_nov_data.dataloader import polycraft_dataloaders_gcd
import polycraft_nov_data.data_const as data_const

from polycraft_nov_det.data.loader_trans import DINOConsistentTrans
from polycraft_nov_det.model_load import load_dino_pretrained
from polycraft_nov_det.models.dino_train import DinoWithHead
import polycraft_nov_det.train as train

# construct argument parser
parser = argparse.ArgumentParser(description="Polycraft Novelty Detection Model Training")
# add args
parser.add_argument("-model", choices=["gcd"],
                    default="gcd", help="Model to train")
parser.add_argument("-name", default=None, help="Name for model run")
parser.add_argument("-gpu", type=int, default=1,
                    help="Index of GPU to train on, negative int for CPU")
parser.add_argument("-sup_weight", type=float, default=.35,
                    help="Hyperparameter for GCD loss")
args = parser.parse_args()
# process args
if args.gpu < 0:
    args.gpu = None

if args.model == "gcd":
    batch_size = 64
    labeled_loader, unlabeled_loader = polycraft_dataloaders_gcd(DINOConsistentTrans(), batch_size)
    # get model instance
    model = DinoWithHead(load_dino_pretrained())
    # start model training
    model_label = args.name if args.name is not None else train.model_label(model, None)
    train.train_gcd(model, model_label, labeled_loader, unlabeled_loader, data_const.NORMAL_CLASSES,
                    gpu=args.gpu, supervised_weight=args.sup_weight)
