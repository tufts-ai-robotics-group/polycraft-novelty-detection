from polycraft_nov_det.eval.eval import eval_mnist, eval_polycraft


def def_eval_mnist():
    eval_mnist("./models/mnist/class_0-4/noisy_500_lr_1e-2/500.pt")


def def_eval_polycraft():
    eval_polycraft("./models/polycraft/noisy/scale_0_75/8000.pt", image_scale=.75, device="cuda:1")
