import torch
import pietorch
import numpy as np
import skimage.exposure
from scipy.ndimage import binary_dilation


def _histogram_equalization(self, reconstructed, voided_modality):
    reconstructed_np = reconstructed.cpu().numpy()  # (B, 1, 240, 240, 155)
    voided_modality_np = voided_modality.cpu().numpy()  # (B, 1, 240, 240, 155)

    reconstructed_he = []
    for i in range(reconstructed_np.shape[0]):
        threshold = 0.01
        reconstructed_flat = reconstructed_np[i, 0][reconstructed_np[i, 0] > threshold]
        original_voided_flat = voided_modality_np[i, 0][voided_modality_np[i, 0] > threshold]

        reconstructed_he_flat = skimage.exposure.match_histograms(
            reconstructed_flat,
            original_voided_flat
        )

        reconstructed_he_matched = reconstructed_np[i, 0].copy()
        mask = reconstructed_np[i, 0] > threshold
        reconstructed_he_matched[mask] = reconstructed_he_flat
        reconstructed_he_matched[~mask] = 0

        reconstructed_he.append(reconstructed_he_matched)

    reconstructed_he = np.stack(reconstructed_he, axis=0)
    reconstructed_he = torch.from_numpy(reconstructed_he).unsqueeze(1).type_as(voided_modality)

    return reconstructed_he  # (B, 1, 240, 240, 155)


def _poisson_blending(self, reconstructed, voided_modality, original_mask):
    reconstructed_pb = reconstructed.clone()  # (B, 1, 240, 240, 155)
    for i in range(reconstructed.shape[0]):
        reconstructed_ = reconstructed[i, 0].to('cpu')
        original_voided_ = (reconstructed * original_mask + voided_modality * (1 - original_mask))[i, 0].to('cpu')
        original_mask_ = original_mask[i, 0].to('cpu')
        corner_coord = torch.tensor([0, 0, 0]).to('cpu')
        reconstructed_pb_ = pietorch.blend(
            source = reconstructed_,
            target = original_voided_,
            mask = original_mask_,
            corner_coord = corner_coord,
            mix_gradients = True,
        )
        reconstructed_pb[i, 0] = reconstructed_pb_  
        
    return reconstructed_pb  # (B, 1, 240, 240, 155)


def _pixel_injection(self, reconstructed, voided_modality, original_mask):
    # original_mask_np = original_mask.cpu().numpy()
    # dilated_mask_np = binary_dilation(original_mask_np, iterations=1)
    # dilated_mask = torch.from_numpy(dilated_mask_np).to(original_mask.device).float().clamp(0, 1)
    dilated_mask = original_mask  # TODO no dilation?

    reconstructed_pi = reconstructed * dilated_mask + voided_modality * (1 - dilated_mask)
    reconstructed_pi[reconstructed_pi < 0.01] = 0

    return reconstructed_pi  # (B, 1, 240, 240, 155)