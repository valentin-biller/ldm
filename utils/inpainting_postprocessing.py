from data import create_conditioning

def _generate_conditioning(self, paths_original_t1_voided, paths_original_mask):
    from gbm_bench.preprocessing.preprocess import preprocess_nifti

    temp_dir = Path(self.dir_output_model) / f"temp_{random.randint(10000, 99999)}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    original_conditionings = []
    for i in range(len(paths_original_t1_voided)):
        path_original_t1_voided = Path(paths_original_t1_voided[i])
        path_original_mask = Path(paths_original_mask[i])

        temp_patient = path_original_t1_voided.parent.name
        temp_dir_patient = temp_dir / temp_patient
        temp_dir_patient.mkdir(parents=True, exist_ok=True)

        path_inverted_mask = temp_dir_patient / f"{temp_patient}-mask-inverted.nii.gz"
        image_original_mask = nib.load(path_original_mask)
        data_original_mask = image_original_mask.get_fdata()
        affine_original_mask = image_original_mask.affine
        inverted_mask = (data_original_mask == 0).astype(np.float32)
        nib.save(nib.Nifti1Image(inverted_mask, affine_original_mask), path_inverted_mask)

        if torch.cuda.is_available():
            device_str = str(torch.cuda.current_device())
        else:
            device_str = 'cpu'

        preprocess_nifti(
            t1_file=path_original_t1_voided,
            t1c_file='.',
            t2_file='.',
            flair_file='.',
            pre_treatment=True,
            outdir=temp_dir_patient,
            is_coregistered=True,
            is_skull_stripped=True,
            # tumorseg_file=Path(temp_dir),
            cuda_device=device_str,
            registration_mask_file=path_inverted_mask
        )

        path_original_tissue_segmentation = temp_dir_patient / 'processed' / 'tissue_segmentation' / 'tissue_seg.nii.gz'
        original_tissue_segmentation = nib.load(path_original_tissue_segmentation).get_fdata()  # 240, 240, 155
        original_tissue_segmentation = torch.as_tensor(original_tissue_segmentation).float()
        original_growth_model = torch.zeros_like(original_tissue_segmentation)
        original_conditioning = create_conditioning(original_growth_model, original_tissue_segmentation)
        original_conditioning = torch.as_tensor(original_conditioning).float()  # 4, 240, 240, 155

        original_conditionings.append(original_conditioning)

    shutil.rmtree(temp_dir)

    return torch.stack(original_conditionings, dim=0).to(self.device)

def _histogram_equalization(self, reconstructed_t1, original_t1_voided):

    reconstructed_t1_np = reconstructed_t1.cpu().numpy()  # (B, 1, 240, 240, 155)
    original_t1_voided_np = original_t1_voided.cpu().numpy()  # (B, 1, 240, 240, 155)

    reconstructed_t1_he = []
    for i in range(reconstructed_t1_np.shape[0]):
        threshold = 10
        reconstructed_t1_flat = reconstructed_t1_np[i, 0][reconstructed_t1_np[i, 0] > threshold]
        original_t1_voided_flat = original_t1_voided_np[i, 0][original_t1_voided_np[i, 0] > threshold]

        reconstructed_t1_he_flat = skimage.exposure.match_histograms(
            reconstructed_t1_flat,
            original_t1_voided_flat
        )

        reconstructed_t1_he_matched = reconstructed_t1_np[i, 0].copy()
        mask = reconstructed_t1_np[i, 0] > threshold
        reconstructed_t1_he_matched[mask] = reconstructed_t1_he_flat
        reconstructed_t1_he_matched[~mask] = 0

        reconstructed_t1_he.append(reconstructed_t1_he_matched)

    reconstructed_t1_he = np.stack(reconstructed_t1_he, axis=0)
    reconstructed_t1_he = torch.from_numpy(reconstructed_t1_he).unsqueeze(1).type_as(original_t1_voided)

    return reconstructed_t1_he  # (B, 1, 240, 240, 155)

def _poisson_blending(self, reconstructed_t1, original_t1_voided, original_mask):
    reconstructed_t1_pb = reconstructed_t1.clone()  # (B, 1, 240, 240, 155)
    for i in range(reconstructed_t1.shape[0]):
        reconstructed_t1_ = reconstructed_t1[i, 0].to('cpu')
        original_t1_voided_ = (reconstructed_t1 * original_mask + original_t1_voided * (1 - original_mask))[i, 0].to('cpu')
        original_mask_ = original_mask[i, 0].to('cpu')
        corner_coord = torch.tensor([0, 0, 0]).to('cpu')
        reconstructed_t1_pb_ = pietorch.blend(
            source = reconstructed_t1_,
            target = original_t1_voided_,
            mask = original_mask_,
            corner_coord = corner_coord,
            mix_gradients = True,
        )
        reconstructed_t1_pb[i, 0] = reconstructed_t1_pb_  
        
    return reconstructed_t1_pb  # (B, 1, 240, 240, 155)

def _pixel_injection(self, reconstructed_t1, original_t1_voided, original_mask):
    original_mask_np = original_mask.cpu().numpy()
    dilated_mask_np = binary_dilation(original_mask_np, iterations=1)
    dilated_mask = torch.from_numpy(dilated_mask_np).to(original_mask.device).float().clamp(0, 1)

    reconstructed_t1_pi = reconstructed_t1 * dilated_mask + original_t1_voided * (1 - dilated_mask)
    reconstructed_t1_pi[reconstructed_t1_pi < 0.01] = 0

    return reconstructed_t1_pi  # (B, 1, 240, 240, 155)