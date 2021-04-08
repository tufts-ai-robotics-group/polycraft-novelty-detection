import torch
import matplotlib.pyplot as plt
import numpy as np

from polycraft_nov_data.polycraft_dataloader import create_data_generators
from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst as LSANet
from polycraft_nov_det import novelty


def check_image_for_novelty(models_state_dict_path):

    # batch size depends on scale we use 0.25 --> 6, 0.5 --> 42, 0.75 --> 110, 1 --> 195
    scale = 0.75
    batch_size = 110

    # get dataloaders
    print('Download zipped files (if necessary), extract the ones we want to have and generate dataloaders.')
    train_loader, valid_loader, test_loader = create_data_generators(
        shuffle=False,
        novelty_type='normal',
        scale_level=scale)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pc_input_shape = (3, 32, 32)  # color channels, height, width

    # construct model
    model = LSANet(pc_input_shape, batch_size)

    # I compute it on a cpu, we might change this later.
    model.load_state_dict(torch.load(models_state_dict_path, map_location=torch.device('cpu')))
    model.eval()

    # ecdf computed for images with no novelty (we should probably compute it for train-, valid and test dataset)
    ecdf = novelty.reconstruction_ecdf_polycraft(model, train_loader)
    rec = novelty.ReconstructionDet(model, ecdf)

    # We will have images from the game here later
    train_loader_nov, valid_loader_nov, test_loader_nov = create_data_generators(
        shuffle=False,
        novelty_type='novel_item', item_to_include='minecraft:tnt', scale_level=scale)

    for i, sample in enumerate(train_loader_nov):

        # sample contains all patches of one screen image and its novelty description
        patches = sample[0]
        nov_dic = sample[1]

        # Dimensions of flat_patches: [1, batch_size, color channels, height of patch, width of patch]
        flat_patches = torch.flatten(patches, start_dim=1, end_dim=2)

        x = flat_patches[0].float().to(device)

        novelty_score = rec.is_novel(x, 0.88)

        fig1 = plt.figure(dpi=1200)

        # Plot input
        for r in range(1, x.shape[0] + 1):

            fig1.add_subplot(patches.shape[1], patches.shape[2], r)
            img = x[r-1, :, :, :]

            fig1.gca().set_title(novelty_score[r-1], fontsize=4)

            plt.subplots_adjust(hspace=1)
            plt.imshow(np.transpose((img).detach().numpy(), (1, 2, 0)))
            plt.tick_params(top=False, bottom=False, left=False,
                            right=False, labelleft=False, labelbottom=False)

        fig1.suptitle('Input image with novelty label')
        plt.show()


if __name__ == '__main__':

    state_dict_path = '../models/polycraft/saved_statedict_polycraft_scale_0_75/LSA_polycraft_no_est_075_980.pt'

    check_image_for_novelty(state_dict_path)
