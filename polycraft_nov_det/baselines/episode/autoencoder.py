if __name__ == "__main__":
    from pathlib import Path

    import torch

    from polycraft_nov_data.dataloader import collate_patches, episode_dataloader
    from polycraft_nov_data.image_transforms import PatchTestPreprocess
    from polycraft_nov_data.novelcraft_const import PATCH_SHAPE

    from polycraft_nov_det.baselines.eval_episode import save_scores, eval_from_save
    from polycraft_nov_det.baselines.novelcraft.autoencoder import ReconstructDetectorPatchBased

    device = torch.device("cuda:0")
    output_folder = Path("models/episode/eval_patch/")
    model_path = Path("models/episode/ae_patch/8000.pt")
    detector = ReconstructDetectorPatchBased(model_path, PATCH_SHAPE, device)
    test_loader = episode_dataloader("test", PatchTestPreprocess(), 1, collate_patches)

    save_scores(detector, output_folder, test_loader)
    eval_from_save(output_folder)
