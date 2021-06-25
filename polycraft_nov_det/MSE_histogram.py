import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
import polycraft_nov_data.data_const as data_const
import polycraft_nov_data.image_transforms as image_transforms


def parse_roi(roi_str):
    if type(roi_str) is not str:
        return None
    parts = roi_str.split(',')
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def roi_coords_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None, sep=';')
    rois = []
    for _, row in df.iterrows():
        roi = parse_roi(row[1])
        rois.append(roi)
    return rois


class ROIPatchSampler(Dataset):
    """
    This patch sampler extracts a pair of image patches, one patch is
    extracted randomly from a region with a novelty, the other from a non-novel
    region.
    The novel region is specified in a csv file by the x- and y-coordinates
    of the upper left corner and its width and height.
    """

    def __init__(self, epochs, scale, size, root_dir, transform=None):
        self.epochs = epochs
        self.scale = scale
        self.size = size
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = int(data_const.PATCH_SHAPE[1] * self.scale**(-1))

    def __len__(self):
        return 20  # We have 20 images per size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_dir = str(idx) + '.png'
        image_dir = os.path.join(self.root_dir, image_dir)
        image = io.imread(image_dir)
        image = self.transform(image)  # minecraft bar is removed here
        image = image.permute(1, 2, 0)

        # image patch with/without novelty, patch size set according to scale
        img_patch_no, img_patch_nono = self.get_patches_from_rois(image, idx)

        img_patch_no = self.rescale_patch(img_patch_no)
        img_patch_nono = self.rescale_patch(img_patch_nono)

        return img_patch_no, img_patch_nono

    def get_patches_from_rois(self, image, idx):
        # Use epoch number as random seed in order to sample different patches
        rng = np.random.RandomState(self.epochs)
        H, W, C = image.shape

        # bounding box values:
        # --> x & y coordinate of upper left corner, its width and height
        roi_coords = self.get_roi_coordinates()
        x_idx, y_idx, width, height = roi_coords[idx]

        sec_marg = int(self.patch_size/2)+1  # security margin
        X_min = sec_marg
        X_max = W - sec_marg
        Y_min = sec_marg
        Y_max = H - sec_marg
        dx = self.patch_size
        dy = self.patch_size

        max_n = 1000
        offset1 = 4 * ((4 - self.size + 1)/2)  # just to make sure that the novelty is more centered
        # just to make sure that we do not capture a novelty here
        offset2 = 4 * ((4 - self.size + 1)/2)

        # -- novelty patch ----------------------------------------------------
        X, Y = 0, 0
        n = 0

        while(X < X_min or X > X_max or Y < Y_min or Y > Y_max):
            # coordinates of center pixel, offset1 to get a more central region
            Y = rng.randint(y_idx + offset1, y_idx + height - offset1)
            X = rng.randint(x_idx + offset1, x_idx + width - offset1)
            n += 1

            if (n > max_n):
                print('No novel region where a whole patch fits in!')
                break

        # patch containing novelty
        novelty_patch = image[Y-int(dy/2):Y+int(dy/2), X-int(dx/2):X+int(dx/2), :]

        # -- no-novelty patch ------------------------------------------------
        n = 0
        while ((X > x_idx-offset2 and X < (x_idx + width + offset2)) or
               (Y > y_idx-offset2 and Y < (y_idx + height+offset2))):
            X = rng.randint(X_min, X_max)
            Y = rng.randint(Y_min, Y_max)
            n += 1
            if (n > max_n):
                print('No non-novel region where a whole patch fits in!')
                break

        no_novelty_patch = image[Y-int(dy/2):Y+int(dy/2), X-int(dx/2):X+int(dx/2), :]

        return novelty_patch, no_novelty_patch

    def rescale_patch(self, patch):
        patch = patch.permute(2, 0, 1)
        p_h, p_w = data_const.PATCH_SHAPE[1],  data_const.PATCH_SHAPE[2]
        return transforms.functional.resize(patch, (p_h, p_w))

    def get_roi_coordinates(self):
        rois = roi_coords_from_csv(self.root_dir + '/size' + str(self.size) + '.csv')
        return rois


def plot_loss_his(scale, size, model_path, epochs):
    """Plots a histogram of reconstruction losses of novel and non-novel
    patches. The plotted loss values are computed based on the MSE loss
    averaged over a patch.
    Args:
        scale: Resolution scaling factor (1 --> original resolution)
        size: Size of the novelty, 0 for largest, 4 for smalles size of novelty
        root_dir: Directory of the images and csv file
        epochs: Number of randomly selected novel/non-novel patch-pairs per
                image
    """
    plt.rcParams['figure.dpi'] = 1200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    pc_input_shape = data_const.PATCH_SHAPE
    n_z = 100
    root_dir = 'datasets/novelty_evaluation_size/' + str(size) + '_/' + str(size)

    trnfrm = transforms.Compose([transforms.ToTensor(), image_transforms.CropUI()])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # construct model
    model = LSANet(pc_input_shape, n_z)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    loss_func = nn.MSELoss()
    all_loss_no = []
    all_loss_nono = []

    for epoch in range(epochs):
        data = ROIPatchSampler(epoch, scale=scale, size=size, root_dir=root_dir,
                               transform=trnfrm)
        train_loader = DataLoader(data, batch_size=1, shuffle=False, **kwargs)

        for i, sample in enumerate(train_loader):

            patch_no = sample[0]  # patch with novelty
            patch_nono = sample[1]  # patch without any novelty

            x_nono = patch_nono
            x_rec_nono, z = model(x_nono.float().to(device))
            x_no = patch_no
            x_rec_no, z = model(x_no.float().to(device))

            # MSE loss, averaged over patch
            loss_nono = loss_func(x_nono, x_rec_nono)
            loss_no = loss_func(x_no, x_rec_no)
            all_loss_nono.append(loss_nono.item())
            all_loss_no.append(loss_no.item())

    plt.hist([all_loss_nono, all_loss_no], bins=140, alpha=0.7,
             color=("b", "m"), label=['no novelty', 'novelty'])
    plt.legend()
    plt.title('Patch-wise MSE loss, size %s, scale %s' % (str(size), str(scale)))
    plt.yticks([])
    plt.xlim([-0.0005, 0.1])

    return


if __name__ == '__main__':

    scale = 0.75
    size = 1
    noe = 20  # Number of randomly selected patch pairs per image
    model_path = 'models/polycraft/no_noise/scale_0_75/7200.pt'

    plot_loss_his(scale, size, model_path, epochs=noe)
