import argparse

import polycraft_nov_det.models.lsa.LSA_cifar10_no_est as LSA_cifar10_no_est
import polycraft_nov_det.models.lsa.LSA_mnist_no_est as LSA_mnist_no_est


# construct argument parser
parser = argparse.ArgumentParser(description="Polycraft Novelty Detection Model Training")
parser.add_argument("model", choices=["mnist", "polycraft"],
                    help="Model to train")
parser.add_argument("-no_noise", action="store_true",
                    help="Don't add noise to training images")
parser.add_argument("-add_novel", action="store_true",
                    help="Include novel images in training")
args = parser.parse_args()

# handle MNIST args
if args.model == "mnist":
    if args.add_novel:
        include_classes = None
    else:
        include_classes = [0, 1, 2, 3, 4]
    LSA_mnist_no_est.train(include_classes=include_classes, train_noisy=not args.no_noise)
# handle Polycraft args
else:
    LSA_cifar10_no_est.train()
