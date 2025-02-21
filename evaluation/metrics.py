import torch
import pandas as pd

def compute_metrics(dsb, dl, n_ipf, num_iter, fb = 'f'):
    from skimage.metrics import mean_squared_error as mse
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    dsb.load_checkpoints(n_ipf, num_iter, fb='b')

    df_dict = {
        "participant_id": [],
        "session_id": [],
        "slice_id": [],
        "metric": [],
        "image_X": [],
        "value": [],
    }
    metrics = {
        "MSE": mse,
        "SSIM": ssim,
        "PSNR": psnr,
    }
    references = {'image': 'input', 'label': 'ground_truth'}

    for i, batch in enumerate(dl):
        batch_size = batch['image'].shape[0]

        with torch.no_grad():
            samples = dsb.sample_batch(batch['image'], fb)
        samples = samples[:, -1, :, :, :] # get the final images

        for i in range(batch_size):
            for ref in references.keys():
                for metric, fn in metrics.items():
                    df_dict["participant_id"].append(batch["participant_id"][i])
                    df_dict["session_id"].append(batch["session_id"][i])
                    df_dict["slice_id"].append(batch["slice_id"][i].item())
                    df_dict["metric"].append(metric)
                    df_dict["image_X"].append(references[ref])
                    if metric == "PSNR":
                        kwargs = {"data_range": 2}
                    elif metric == "SSIM":
                        kwargs = {"data_range": 2}
                    else:
                        kwargs = {}
                    df_dict["value"].append(
                        fn(
                            batch[ref][i].squeeze().cpu().numpy(),
                            samples[i].squeeze().cpu().numpy(),
                            **kwargs
                        ),
                    )

    df = pd.DataFrame(df_dict)

    tsv_dir = dsb.experiment_directory / "evaluation" / "validation"
    tsv_dir.mkdir(parents = True, exist_ok = True)
    tsv_path = tsv_dir / f"{fb}_{n_ipf}_{num_iter}.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    
    return tsv_path
