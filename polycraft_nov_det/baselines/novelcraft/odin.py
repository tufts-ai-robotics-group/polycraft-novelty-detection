import torch
from torchvision.transforms import Normalize

from polycraft_nov_det.detector import NoveltyDetector
from polycraft_nov_det.model_utils import load_vgg_model


class OdinDetector(NoveltyDetector):
    def __init__(self, model_path, device="cpu", temp=1000, noise=.0004):
        super().__init__(device)
        self.model = load_vgg_model(model_path, device).to(device).eval()
        self.temp = temp
        self.noise = noise

    def novelty_score(self, data):
        data = data.to(self.device)
        data.requires_grad = True
        # get model output
        output = self.model(data)
        # get prediction from non-temp scaled output
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        # perturbation from backprop temp scaled CE
        loss = torch.nn.CrossEntropyLoss()(output / self.temp, pred)
        loss.backward()
        # normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # normalizing the gradient to the same space of image
        gradient = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(gradient)
        # adding small perturbations to images
        pert_data = torch.add(data.data, gradient, alpha=-self.noise)
        with torch.no_grad():
            output = self.model(pert_data)
        # score from perturbed image, multiplying by -1 so large scores are more likely novel
        # selecting first element is only to discard argmax from torch.max
        return -1 * torch.max(torch.softmax(output / self.temp, dim=1), dim=1)[0]


if __name__ == "__main__":
    from pathlib import Path

    from polycraft_nov_data.dataloader import novelcraft_dataloader
    from polycraft_nov_data.image_transforms import VGGPreprocess

    from polycraft_nov_det.baselines.eval_novelcraft import save_scores, eval_from_save

    output_parent = Path("models/vgg/eval_odin")
    
    use_novelcraft_plus = True
    
    if use_novelcraft_plus:
        model_path = Path("models/vgg/vgg_classifier_1000_plus.pt")
    
    else:
        model_path = Path("models/vgg/vgg_classifier_1000.pt")
        
    # define hyperparameter search space
    temps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    noises = [i * 0.004 / 20 for i in range(21)]
    for temp in temps:
        for noise in noises:
            output_folder = output_parent / Path(f"t={temp}_n={noise:.4f}")
            save_scores(
                OdinDetector(model_path, device=torch.device("cuda:0"), temp=temp, noise=noise),
                output_folder,
                novelcraft_dataloader("valid", VGGPreprocess(), 32),
                novelcraft_dataloader("test", VGGPreprocess(), 32))
            eval_from_save(output_folder)
