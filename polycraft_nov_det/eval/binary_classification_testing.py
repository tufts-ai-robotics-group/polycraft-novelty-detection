from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset
import polycraft_nov_det.eval.plot as eval_plot
import polycraft_nov_det.eval.stats as eval_stats


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()       
        self.ll1 = nn.Linear(3, 12) 
        self.ll2 = nn.Linear(12, 12)
        self.llout = nn.Linear(12, 1) 
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        #self.n1 = nn.InstanceNorm1d(12)
        #self.n2 = nn.InstanceNorm1d(12)
        
    def forward(self, inputs):
        linear1 = self.relu(self.ll1(inputs))
        linear2 = self.relu(self.ll2(linear1))
        output = self.sigmoid(self.llout(linear2))
    
        return output
        

def loss_vector_evaluation(model_paths):
    
    epochs = 1
    b_s = 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = binaryClassification()
    bc_path = 'models/polycraft/binary_classification/threshold_selection_270.pt'
    classifier.load_state_dict(torch.load(bc_path))
    classifier.eval()
    classifier.to(device)
    BCEloss = nn.BCELoss()
   
    model_path05 = Path(model_paths[0])
    model05 = model_utils.load_polycraft_model(model_path05, device).eval()
    model_path075 = Path(model_paths[1])
    model075 = model_utils.load_polycraft_model(model_path075, device).eval()
    model_path1 = Path(model_paths[2])
    model1 = model_utils.load_polycraft_model(model_path1, device).eval()

    _, valid_loader05, test_loader05 = polycraft_dataloaders(batch_size=b_s,
                                                            image_scale=0.5, 
                                                            include_novel=True, 
                                                            shuffle=True)
    _, valid_loader075, test_loader075 = polycraft_dataloaders(batch_size=b_s,
                                                            image_scale=0.75, 
                                                            include_novel=True, 
                                                            shuffle=True)
    _, valid_loader1, test_loader1 = polycraft_dataloaders(batch_size=b_s,
                                                            image_scale=1, 
                                                            include_novel=True, 
                                                            shuffle=True)
    
    # get targets determined at runtime
    base_dataset = polycraft_dataset()                                         
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]
    
    predictions = []
    labels = []  # label 0 --> height, label 1 --> items, label 2 --> no novelty
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    
    # train model
    for epoch in range(epochs):
       
        test_loss = 0
    
        for i, samples in enumerate(zip(test_loader05, test_loader075, 
                                        test_loader1)):
            
            loss_vector = []
            
            with torch.no_grad():
                
                for n, model in enumerate(all_models):
                        
                    patches = samples[n][0]
                        
                    _, ih, iw = polycraft_const.IMAGE_SHAPE
                    _, ph, pw = polycraft_const.PATCH_SHAPE
                        
                    x = patches.float().to(device)
                    x.requires_grad = False
                    x_rec, z = model(x)
                    
                    loss2d = rec_loss2d(x_rec, x)
                    loss2d = torch.mean(loss2d, (1, 2, 3))  # avgd. per patch
                    maxloss = torch.max(loss2d)
                    loss_vector.append(maxloss.item())  
                    
            label = samples[0][1]  
            
            if label == 0 or label == 1:
                label = True
            if label == 2:
                label = False
                
            labels.append(label)
                
            pred = classifier(torch.FloatTensor(loss_vector).to(device))
            
            if pred >= 0.5 and label == True:
                tp += 1
            if pred >= 0.5 and label == False:
                fp += 1
            if pred < 0.5 and label == False:
                tn += 1
            if pred < 0.5 and label == True:
                fn += 1
                
    del base_dataset
    return tp, fp, tn, fn
      

if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    tp, fp, tn, fn = loss_vector_evaluation(paths)
    cm = np.array([[tp, fn],
                   [fp, tn]])
    eval_plot.plot_con_matrix(cm).savefig(("con_matrix_ms.png"))

  
