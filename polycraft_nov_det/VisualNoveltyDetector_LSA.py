import torch
import torch.nn as nn
from skimage.transform import rescale

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from data_handler import read_image_csv

def normalize_img(img):
    minval = img.min()
    maxval = img.max()
    
    diff = maxval - minval
    
    if diff > 0:
        img_norm = (img - minval) / diff
    else:
        img_norm = torch.zeros(img.size())
    
    return img_norm


class VisualNoveltyDetector:
    def __init__(self, state_dict_path, scale_factor):
    
        pc_input_shape = (3, 32, 32)  # color channels, height, width of one patch
        n_z = 110
     
        # construct model
        model = LSANet(pc_input_shape, n_z)
        
        #I compute it on a cpu, we might change this later!?
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu'))) 
        model.eval()
        
        self.model = model
        self.scale_factor = scale_factor
        self.p_size = 32
        
    def crop_rescale_normalize(self, image):
        
        image = image[0:234, :, :] #remove minecraft bar
        image = rescale(image, (self.scale_factor, self.scale_factor, 1), anti_aliasing = True)
        image = normalize_img(image) 
        
        return image
    
    def extract_patches(self, image):
    
        #Extract patches
        stride = int(self.p_size/2) # patch stride
        image = torch.from_numpy(image)
        patches = image.unfold(0, self.p_size, stride).unfold(1, self.p_size, stride)
        
        return torch.flatten(patches, start_dim=0, end_dim=1)   
   
    def apply_model_and_compute_MSE(self, img):
    
        MSEloss = nn.MSELoss()
    
        img = self.crop_rescale_normalize(img)
        x = self.extract_patches(img)

        x_rec, z = self.model(x.float())
        
        rec_loss = MSEloss(x, x_rec)

        return rec_loss

    def check_for_novelty(self, img):#, state_json = None, get_object= False, include_positions = False, score_th = 0, scoring_algorithm = "GMM"):

        print('We compute a score here, then we send it back!')
        
        img_rec_loss = self.apply_model_and_compute_MSE(img)
        #How exactly do we want to evaluate if an image is novel!?!?
        #res = ???
        
        return #res
        
        
if __name__== "__main__":
    #Test on csv from polycraft which includes the save_screen json!
    data, states = read_image_csv(path = "novelty-lvl-1_2021-04-12-10-44-43_SENSE-SCREEN.csv", n_images = None, load_states = False)
    detector = VisualNoveltyDetector(state_dict_path= 'saved_statedict/LSA_polycraft_no_est_075_980.pt', scale_factor=0.75)
    detector.check_for_novelty(data[0])

