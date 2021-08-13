import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from polycraft_nov_data.dataloader import polycraft_dataloaders


class SimpleAutoencoder(nn.Module):
    
    def __init__(self, in_feat, lat_dim, useBias=False):
        super(SimpleAutoencoder, self).__init__()     
        
        self.encoder = torch.nn.Linear(in_features=in_feat, out_features=lat_dim, bias=useBias)  
        self.decoder = torch.nn.Linear(in_features=lat_dim, out_features=in_feat, bias=useBias)
        
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax( dim = 1)
        
    
    def forward(self, x):
        
        z = self.leakyrelu( self.encoder(x) )
        x_rec = self.relu( self.decoder(z) )
       
        return x_rec
    

def learn_histograms():
    """
    Train model on histograms of grayscale patches.
    """
    
    # get a unique path for this session to prevent overwriting
    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    
     # get Tensorboard writer
    writer = SummaryWriter("runs_histogram_0_75_nz16/"  + start_time)

    scale = 0.75
    epochs = 4000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleAutoencoder(256, 16)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    
    train_loader, valid_loader, _ = polycraft_dataloaders(batch_size=1,
                                                            image_scale=scale, 
                                                            include_novel=False, 
                                                            shuffle=False)
    print('Size of training loader', len(train_loader), flush=True)
    
    for epoch in range(epochs):
        print('---------- Epoch ', epoch, ' -------------', flush=True)
        train_loss = 0
        for i, sample in enumerate(train_loader):
           
            patch = sample[0]
            batch_size = patch.shape[0]
            histos = []
            
            for p in range(patch.shape[0]):
                x_p = np.transpose(patch[p].detach().numpy(), (1, 2, 0))
                x_com = cv2.normalize(src=x_p, dst=None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                x_com  = cv2.cvtColor(x_com,cv2.COLOR_RGB2GRAY)
                
              
                histogram, bin_edges = np.histogram(
                    x_com[:, :], bins=256, range=(0, 256))
                histos.append(histogram)
                
            histos = torch.FloatTensor(histos) 
            
            data = histos.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            r_data = model(data) 
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            optimizer.step()
            
            # logging
            train_loss += batch_loss.item() * batch_size
        # calculate and record train loss
        av_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Average Train Loss", av_train_loss, epoch)
        print('Average training loss  ', av_train_loss, flush=True)
        
        valid_loss = 0    
        for i, sample in enumerate(valid_loader):
           
            patch = sample[0]
            batch_size = patch.shape[0]
            histos = []
            
            for p in range(patch.shape[0]):
                x_p = np.transpose(patch[p].detach().numpy(), (1, 2, 0))
                x_com = cv2.normalize(src=x_p, dst=None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                x_com  = cv2.cvtColor(x_com,cv2.COLOR_RGB2GRAY)
                
              
                histogram, bin_edges = np.histogram(
                    x_com[:, :], bins=256, range=(0, 256))
                histos.append(histogram)
                
            histos = torch.FloatTensor(histos) 
            
            data = histos.to(device)
            r_data = model(data) 
            batch_loss = loss_func(data, r_data)
            
            valid_loss += batch_loss.item() * batch_size
        av_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar("Average Validation Loss", av_valid_loss, epoch)
        print('Average validation loss  ', av_valid_loss, flush=True)
        # updates every 10% of training time
        if (epochs >= 10 and (epoch + 1) % (epochs // 10) == 0) or epoch == epochs - 1:
           
            # save model
            torch.save(model.state_dict(), "saved_statedict/histogram_075_nz50_%d.pt" % (epoch + 1,))
    return model

            
            
if __name__ == '__main__':
    
    ae = learn_histograms()

    
    
