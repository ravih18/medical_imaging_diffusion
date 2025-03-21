import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from evaluation.mask import get_mni_mask, get_pathology_mask
from evaluation.healthiness import healthiness_score

def mse_in_mask(X, Y, mask):
    return ((X - Y)**2).mean(where=mask.astype(bool))


def psnr_in_mask(X, Y, data_range, mask):
    mse = mse_in_mask(X, Y, mask)
    return 10 * np.log10((data_range**2) / mse)


def compute_metrics(dsb, dl, n_ipf, num_iter, img_size, fb = 'f'):
    #from skimage.metrics import mean_squared_error as mse
    from skimage.metrics import structural_similarity as ssim
    #from skimage.metrics import peak_signal_noise_ratio as psnr

    dsb.load_checkpoints(n_ipf, num_iter, fb='b')

    df_dict = {
        "participant_id": [],
        "session_id": [],
        "slice_id": [],
        "metric": [],
        "image_X": [],
        "value": [],
    }
    metrics = ["MSE", "SSIM", "PSNR"]
    references = {'image': 'input', 'label': 'ground_truth'}

    df_h_dict = {
        "participant_id": [],
        "session_id": [],
        "slice_id": [],
        "image": [],
        "healthiness": [],
    }

    mni_mask, mask_transform = get_mni_mask(img_size)
    mask_ad, _ = get_pathology_mask('AD', img_size=img_size)
    mask_out_ad = mni_mask - mask_ad # SOUS CONDITION DES DIMENSIONS DE L'IMAGE

    renorm = transforms.Compose([
        transforms.Lambda(lambda t: 0.5*(t+1))
    ])

    for i, batch in enumerate(dl):
        batch_size = batch['image'].shape[0]

        with torch.no_grad():
            samples = dsb.sample_batch(batch['image'], fb)
        samples = samples[:, -1, :, :, :] # get the final images

        for i in range(batch_size):
            slice_id = batch["slice_id"][i].item()

            mask_mni_slice = mask_transform(mni_mask[:,:,:,slice_id]).squeeze().numpy()
            mask_ad_slice = mask_transform(mask_ad[:,:,:,slice_id]).squeeze().numpy()
            mask_out_ad_slice = mask_transform(mask_out_ad[:,:,:,slice_id]).squeeze().numpy()

            Y = renorm(samples[i].cpu().squeeze().numpy())

            for ref in references.keys():

                X = renorm(batch[ref][i].cpu().squeeze().numpy())
                
                for metric in metrics:
                    df_dict["participant_id"].append(batch["participant_id"][i])
                    df_dict["session_id"].append(batch["session_id"][i])
                    df_dict["slice_id"].append(slice_id)
                    df_dict["metric"].append(metric)
                    df_dict["image_X"].append(references[ref])

                    if metric == "PSNR":
                        value = psnr_in_mask(X, Y, data_range=1, mask=mask_mni_slice)
                    elif metric == "SSIM":
                        _, ssim_map = ssim(X, Y, data_range=1, full=True)
                        value = ssim_map.mean(where=mask_mni_slice.astype(bool))
                    elif metric == "MSE":
                        value = mse_in_mask(X, Y, mask_mni_slice)

                    df_dict["value"].append(value)

                df_h_dict["participant_id"].append(batch["participant_id"][i])
                df_h_dict["session_id"].append(batch["session_id"][i])
                df_h_dict["slice_id"].append(slice_id)
                if ref == "image":
                    df_h_dict["image"].append('input')
                    df_h_dict["healthiness"].append(healthiness_score(X, mask_ad_slice, mask_out_ad_slice))
                else:
                    df_h_dict["image"].append('output')
                    df_h_dict["healthiness"].append(healthiness_score(Y, mask_ad_slice, mask_out_ad_slice))
 
    tsv_dir = dsb.experiment_directory / "evaluation" / "validation"
    tsv_dir.mkdir(parents = True, exist_ok = True)

    df = pd.DataFrame(df_dict)
    tsv_metric_path = tsv_dir / f"metrics_{fb}_{n_ipf}_{num_iter}.tsv"
    df.to_csv(tsv_metric_path, sep="\t", index=False)

    df_h = pd.DataFrame(df_h_dict)
    tsv_healthiness_path = tsv_dir / f"healthiness_{fb}_{n_ipf}_{num_iter}.tsv"
    df_h.to_csv(tsv_healthiness_path, sep="\t", index=False)
    
    return tsv_metric_path, tsv_healthiness_path
