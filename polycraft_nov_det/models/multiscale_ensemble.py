from pathlib import Path

import torch

from polycraft_nov_data.image_transforms import TestPreprocess
import polycraft_nov_det.model_utils as model_utils
from torchvision.transforms import functional
from polycraft_nov_data.data_const import PATCH_SHAPE, IMAGE_SHAPE
from torch.nn.functional import mse_loss


class MultiScaleEnsembleModel():
    def __init__(self, img, dev='cpu'):
        self.img = img
        self.dev = dev
    
        
    def forward(self):
        
        all_models, novelty_classifier = self.load_models()
        img_all_scales = self.process_img_allscales()
        rgb_loss_arrays = self.compute_rgb_loss_arrays(img_all_scales)
        score = novelty_classifier(rgb_loss_arrays)
        
        return score
                
    
    def load_models(self):
        
        path_classifier = 'models/polycraft/binary_classification/threshold_selection_conv_v3_rgb_500.pt'
        path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
        path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
        path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
        paths = [path_classifier, path05, path075, path1]
        
        # Use the model trained on rec.-loss error
        bc_path = paths[0]
        classifier = model_utils.load_polycraft_classifier(bc_path, 
                                                           device=self.dev, 
                                                        add_16x16_model=False)
        classifier.to(self.dev)
        
        # Use 3x32x32 patch based model on all three scales (for now)
        model_path05 = Path(paths[1])
        model05 = model_utils.load_polycraft_model(model_path05, 
                                                   self.dev).eval()
        model_path075 = Path(paths[2])
        model075 = model_utils.load_polycraft_model(model_path075, 
                                                    self.dev).eval()
        model_path1 = Path(paths[3])
        model1 = model_utils.load_polycraft_model(model_path1, 
                                                  self.dev).eval()
        
        return [model05.to(self.dev), model075.to(self.dev), 
                model1.to(self.dev)], classifier
        
       
    def process_img_allscales(self):
        
        processing05 = TestPreprocess(image_scale=.5)
        processing075 = TestPreprocess(image_scale=.75)
        processing1 = TestPreprocess(image_scale=1)
        input05 = processing05(self.img)
        input05 = torch.reshape(input05, (-1,) + input05.shape[2:])
        input075 = processing075(self.img)
        input075 = torch.reshape(input075, (-1,) + input075.shape[2:])
        input1 = processing1(self.img)
        input1 = torch.reshape(input1, (-1,) + input1.shape[2:])
        
        return [input05, input075, input1]
    
    
    def compute_rgb_loss_arrays(self, img_all_scales):
        
        all_models = self.load_models()
        
        # shapes of "patch array" for all scales.
        ipt_shapes = [[6, 7],  # Scale 0.5, patch size 3x32x32
                      [9, 11],  # Scale 0.75, patch size 3x32x32
                      [13, 15]]  # Scale 1, patch size 3x32x32
        
        with torch.no_grad():
            
            loss_arrays = ()

            for n, model in enumerate(all_models):
                
                data = img_all_scales[n].float().to(self.dev)
                data = torch.flatten(data, start_dim=0, end_dim=1)
                _, ih, iw = IMAGE_SHAPE
                _, ph, pw = PATCH_SHAPE
                ipt_shape = ipt_shapes[n]
                
                r_data, z = model(data)
                # #patches x 3 x 32 x 32
                loss2d = mse_loss(data, r_data, reduction="none")
                loss2d = torch.mean(loss2d, (2, 3))  # avgd. per patch
                # Reshape loss values from flattened to "squared shape"
                loss2d_r = loss2d[:, 0].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_g = loss2d[:, 1].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                loss2d_b = loss2d[:, 2].reshape(1, 1, ipt_shape[0], ipt_shape[1])
                # Concatenate to a "3 channel" loss array
                loss2d_rgb = torch.cat((loss2d_r, loss2d_g), 1)
                loss2d_rgb = torch.cat((loss2d_rgb, loss2d_b), 1)
                # Interpolate smaller scales such that they match scale 1
                loss2d = functional.resize(loss2d_rgb, (13, 15))
                loss_arrays = loss_arrays + (loss2d,)
                
        return loss_arrays

