from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from polycraft_nov_data import data_const as polycraft_const
import polycraft_nov_det.model_utils as model_utils
from polycraft_nov_data.dataloader import polycraft_dataset_for_ms, polycraft_dataset
import polycraft_nov_det.models.multiscale_classifier as ms_classifier


class TrippleDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        xC = self.datasetC[index]
        return xA, xB, xC
    
    def __len__(self):
        return len(self.datasetA)


def train_on_loss_vector(model_paths):
    lr = 0.003
    epochs = 300
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    classifier = ms_classifier.MultiscaleClassifierFeatureVector(3)
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

    _, valid_set05, test_set05 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.5, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set075, test_set075 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=0.75, 
                                                            include_novel=True, 
                                                            shuffle=False)
    _, valid_set1, test_set1 = polycraft_dataset_for_ms(batch_size=1,
                                                            image_scale=1, 
                                                            include_novel=True, 
                                                            shuffle=False)
    
    valid_set = TrippleDataset(valid_set05, valid_set075, valid_set1)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    
    test_set = TrippleDataset(test_set05, test_set075, test_set1)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    # get targets determined at runtime
    base_dataset = polycraft_dataset()                                         
    rec_loss2d = nn.MSELoss(reduction='none')
    all_models = [model05.to(device), model075.to(device), model1.to(device)]
    
    # train model
    for epoch in range(epochs):
        print('Epoch number  ', epoch, flush=True)
        train_loss = 0
        valid_loss = 0
        train_acc = 0
        valid_acc = 0
    
        for i, samples in enumerate(valid_loader):
            
            loss_vector = []
            
            with torch.no_grad():
                
                for n, model in enumerate(all_models):
                        
                    patches = samples[n][0][0]
                    patches = torch.flatten(patches, start_dim=0, end_dim=1)
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
            
            if pred >= 0.5 and target.to(device) == torch.ones(1).to(device):
                train_acc += 1
            if pred < 0.5 and target.to(device) == torch.zeros(1).to(device):
                train_acc += 1
         
            # logging
            train_loss += loss.item()
            
        train_loss = train_loss / (len(valid_loader))
        train_acc = train_acc / (len(valid_loader))
        writer.add_scalar("Average Train Loss", train_loss, epoch)
        print('Average training loss  ', train_loss, flush=True)
        writer.add_scalar("Average Train Acc", train_acc, epoch)
        print('Average training Acc  ', train_acc, flush=True)
        
        #print('Prediction', pred, 'Target', target)
            
        for i, samples in enumerate(test_loader):
            
            loss_vector = []
            
            with torch.no_grad():
                
                for n, model in enumerate(all_models):
                        
                    patches = samples[n][0][0]
                    patches = torch.flatten(patches, start_dim=0, end_dim=1)
                        
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
            
            if pred >= 0.5 and target.to(device) == torch.ones(1).to(device):
                valid_acc += 1
            if pred < 0.5 and target.to(device) == torch.zeros(1).to(device):
                valid_acc += 1
            
            # logging
            valid_loss += loss.item()
            
        
        valid_loss = valid_loss / (len(test_loader))
        valid_acc = valid_acc / (len(test_loader))
        writer.add_scalar("Average Validation Loss", valid_loss, epoch)
        writer.add_scalar("Average Validation Acc", valid_acc, epoch)
        print('Average Validation loss  ', valid_loss, flush=True)
        print('Average Validation Acc  ', valid_acc, flush=True)
        
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

   