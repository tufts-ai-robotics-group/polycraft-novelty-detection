from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset


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


def train_on_loss_vector(model_paths):
    lr = 0.003
    epochs = 300
    b_s = 10
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = binaryClassification()
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr)
    BCEloss = nn.BCELoss()
    
    # get Tensorboard writer
    writer = SummaryWriter("runs_binary_class/"  +
                           datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

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
    
    # train model
    for epoch in range(epochs):
        print('Epoch number  ', epoch, flush=True)
        train_loss = 0
        valid_loss = 0
    
        for i, samples in enumerate(zip(valid_loader05, valid_loader075, 
                                        valid_loader1)):
            
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
                target = torch.ones(1).to(device)
            if label == 2:
                target = torch.zeros(1).to(device)
                
            optimizer.zero_grad()
            pred = classifier(torch.FloatTensor(loss_vector).to(device))
            loss = BCEloss(pred, target)
            loss.backward()
            optimizer.step()
         
            # logging
            train_loss += loss.item()*b_s
            
        train_loss = train_loss / (len(valid_loader05))
        writer.add_scalar("Average Train Loss", train_loss, epoch)
        print('Average training loss  ', train_loss, flush=True)
        
        #print('Prediction', pred, 'Target', target)
            
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
                target = torch.ones(1).to(device)
            if label == 2:
                target = torch.zeros(1).to(device)
            
            pred = classifier(torch.FloatTensor(loss_vector).to(device))
            loss = BCEloss(pred, target)
            
            # logging
            valid_loss += loss.item()*b_s 
        
        valid_loss = valid_loss / (len(test_loader05))
        writer.add_scalar("Average Validation Loss", valid_loss, epoch)
        print('Average Validation loss  ', valid_loss, flush=True)
        
        # save model
        if (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            torch.save(classifier.state_dict(), "threshold_selection_%d.pt" % (epoch + 1,))
            
    del base_dataset
    return 
  
        
if __name__ == '__main__':
    path05 = 'models/polycraft/no_noise/scale_0_5/8000.pt'
    path075 = 'models/polycraft/no_noise/scale_0_75/8000.pt'
    path1 = 'models/polycraft/no_noise/scale_1/8000.pt'
    paths = [path05, path075, path1]
    train_on_loss_vector(paths)

   