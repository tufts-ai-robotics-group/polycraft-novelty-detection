import torch
from torchvision.transforms import Normalize

from polycraft_nov_det.detector import NoveltyDetector


class OdinDetector(NoveltyDetector):
    def __init__(self, model, device="cpu", temp=1, noise=.0014):
        super().__init__(device)
        self.model = model.eval().to(device)
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
        pert_data = torch.add(data.data, -self.noise, gradient)
        output = self.model(pert_data)
        # score from perturbed image
        # selecting first element is only to discard argmax from torch.max
        return torch.max(torch.softmax(output / self.temp, dim=1), dim=1)[0]
