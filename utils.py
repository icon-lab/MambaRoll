import os
import warnings
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import matplotlib.pyplot as plt


# TODO: Make sure it works when batch size > 1
def split_mask(mask, ratio):
    # If mask bool, convert to float
    if mask.dtype == torch.bool:
        mask = mask.to(torch.float)

    # Get random indexes from non-zero values
    indices = torch.where(mask.flatten() > 0)[0]
    indices = indices[torch.randperm(indices.numel())[:int(indices.numel()*ratio)]]

    # Create loss mask
    loss_mask = torch.zeros_like(mask)
    loss_mask.view(-1)[indices] = 1

    # Create loss mask
    us_mask = mask - loss_mask

    return loss_mask, us_mask


def fft2c(x, dim=(-2, -1)):
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fft2(x, dim=dim)
    return torch.fft.fftshift(x, dim=dim)


def ifft2c(x, dim=(-2, -1)):
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifft2(x, dim=dim)
    return torch.fft.fftshift(x, dim=dim)


def data_consistency(
    source: torch.complex,
    target: torch.complex,
    mask: torch.tensor
) -> torch.complex:
    """ Fill in k-space data of source image using target image and mask """

    # Fourier transform
    source_kspace = fft2c(source)
    target_kspace = fft2c(target)

    # Fill in source k-space from target k-space
    source_kspace = source_kspace * (1 - mask) + target_kspace * mask

    # Inverse Fourier transform
    source = ifft2c(source_kspace)

    return source


def to_complex(x):
    """ Convert N x (2 x C) x H x W tensor to N x C x H x W complex tensor """
    return torch.complex(x[:, :x.shape[1]//2], x[:, x.shape[1]//2:])


def pad_data(data, image_size):
    """ Pad data to image_size x image_size """
    H, W = data.shape[-2:]

    pad_top = (image_size - H) // 2
    pad_bottom = image_size - H - pad_top
    pad_left = (image_size - W) // 2
    pad_right = image_size - W - pad_left

    return np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))


def save_val_image(target, source, predicted, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_image = min(4, target.shape[0])
    fig, axes = plt.subplots(nrows=4, ncols=n_image, figsize=(n_image*1.5, 6.5))

    if n_image == 1:
        axes = axes[..., None]

    # Dark background
    plt.style.use('dark_background')

    for i in range(n_image):
        # Input image
        axes[0, i].imshow(source[i].squeeze().cpu(), cmap='gray')
        axes[0, i].axis('off')

        # Ground truth image
        axes[1, i].imshow(target[i].squeeze().cpu(), cmap='gray')
        axes[1, i].axis('off')

        # Predicted image
        im = axes[2, i].imshow(predicted[i].squeeze().cpu(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(
            f'PSNR: {metrics["psnrs"][i]:.2f} - SSIM: {metrics["ssims"][i]:.2f}',
            fontsize=5,
            color='white'
        )

        # Diff image
        diff = (norm_01(target[i]) - norm_01(predicted[i])).abs()
        im = axes[3, i].imshow(diff.squeeze().cpu(), cmap='hot')
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=4)
        axes[3, i].axis('off')
        axes[3, i].set_title(
            f'MAE: {diff.mean():.4f}',
            fontsize=5,
            color='white'
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path, bbox_inches='tight', dpi=220)
    plt.close()


def save_eval_images(
    source_images,
    target_images,
    pred_images,
    psnrs,
    ssims,
    save_path
):
    h, w = 30, 30
    zoom_region = [100-w, 100+w, 100-h, 100+h]
    zoom_size = [0, -0.4, 1, 0.47]

    # Squeeze channel dimension
    source_images = source_images.squeeze() if source_images.ndim == 4 else source_images
    target_images = target_images.squeeze() if target_images.ndim == 4 else target_images
    pred_images = pred_images.squeeze() if pred_images.ndim == 4 else pred_images

    # Diff images
    target_norm = norm_01(target_images)
    pred_norm = norm_01(pred_images)
    diff_images = np.abs(target_norm - pred_norm)
    
    plt.style.use('dark_background')

    for i in range(len(source_images)):
        fig, ax = plt.subplots(1, 4, figsize=(18, 16))
        
        ax_zoomed(ax[0], mean_norm(source_images[i]), zoom_region, zoom_size)
        ax_zoomed(ax[1], mean_norm(target_images[i]), zoom_region, zoom_size)
        ax_zoomed(ax[2], mean_norm(pred_images[i]), zoom_region, zoom_size)
        im = ax[3].imshow(diff_images[i], cmap='hot')

        ax[0].set_title('Source')
        ax[1].set_title('Target')
        ax[2].set_title(f'PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.2f}')
        ax[3].set_title(f'Difference\nMAE: {diff_images[i].mean():.4f}')

        # Add colorbar to diff image
        cbar = fig.colorbar(im, fraction=0.046)
        cbar.ax.tick_params(labelsize=8)
        ax[3].axis('off')

        # Save figure
        path = os.path.join(save_path, 'sample_images', f'slice_{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_preds(preds, path):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, preds)


def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)


def norm_01(x):
    if isinstance(x, torch.Tensor):
        min_val = x.amin(axis=(-1,-2), keepdims=True)
        max_val = x.amax(axis=(-1,-2), keepdims=True)
    elif isinstance(x, np.ndarray):
        min_val = x.min(axis=(-1,-2), keepdims=True)
        max_val = x.max(axis=(-1,-2), keepdims=True)

    return (x - min_val)/(max_val - min_val)


def max_norm(x):
    x = x/x.max(axis=(-1,-2), keepdims=True)
    return x

def mean_norm(x):
    x = np.abs(x)
    return x/x.mean(axis=(-1,-2), keepdims=True)


def mean_norm_complex(x):
    x = x / np.abs(x).mean(axis=(-1,-2), keepdims=True)
    return x


def apply_mask_and_norm(x, mask, norm_func):
    x = x*mask
    x = norm_func(x)
    return x


def center_crop(x, crop):
    h, w = x.shape[-2:]
    x = x[..., h//2-crop[0]//2:h//2+crop[0]//2, w//2-crop[1]//2:w//2+crop[1]//2]
    return x


def ax_zoomed(
    ax,
    im,
    zoom_region,
    zoom_size,
    vmin=None,
    vmax=None,
    zoom_edge_color='yellow'
):
    ax.imshow(np.flip(im, axis=0), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    x1, x2, y1, y2 = zoom_region
    axins = ax.inset_axes(
        zoom_size,
        xlim=(x1, x2), ylim=(y1, y2))
    
    axins.imshow(np.flip(im, axis=0), cmap='gray', vmin=vmin, vmax=vmax)

    # Add border to zoomed region
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Remove inset axes ticks/labels
    axins.set_xticks([])
    axins.set_yticks([])
    
    ax.indicate_inset_zoom(axins, edgecolor=zoom_edge_color, linewidth=3)
    ax.axis('off')


def compute_metrics(
    gt_images,
    pred_images, 
    mask=None,
    norm='mean',
    subject_ids=None,
    report_path=None
):
    """ Compute PSNR and SSIM between gt_images and pred_images.
    
    Args:
        gt_images (torch.Tensor): Ground truth images.
        pred_images (torch.Tensor): Predicted images.
        mask (torch.Tensor): Mask to apply to images.
        crop (tuple): Center crop images to (Height, Width).
        norm (str): Normalization method. Options: 'mean', '01'.
        subject_ids (list): List of subject IDs for each slice.

    Returns:
        dict: Dictionary containing PSNR and SSIM values.
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # If images 4-dimensional, squeeze channel dimension
        gt_images = gt_images.squeeze() if gt_images.ndim == 4 else gt_images
        pred_images = pred_images.squeeze() if pred_images.ndim == 4 else pred_images

        # If images 2-dimensional, add channel dimension
        gt_images = gt_images[None, ...] if gt_images.ndim == 2 else gt_images
        pred_images = pred_images[None, ...] if pred_images.ndim == 2 else pred_images

        assert gt_images.shape == pred_images.shape, \
            "Ground truth and predicted images must have the same shape"

        # Compute psnr and ssim
        psnr_values = []
        ssim_values = []

        # Normalize function
        if norm == 'mean':
            norm_func = mean_norm
        elif norm == '01':
            norm_func = norm_01

        # If torch tensor, convert to numpy
        if isinstance(gt_images, torch.Tensor):
            gt_images = gt_images.cpu().numpy()

        if isinstance(pred_images, torch.Tensor):
            pred_images = pred_images.cpu().numpy()

        # If images between [-1, 1], scale to [0, 1]
        if np.nanmin(gt_images) < -0.1:
            gt_images = ((gt_images + 1) / 2).clip(0, 1)

        if np.nanmin(pred_images) < -0.1:
            pred_images = ((pred_images + 1) / 2).clip(0, 1)

        # Apply mask and normalize
        if mask is not None:
            # Crop to mask shape
            gt_images = center_crop(gt_images, mask.shape[-2:])
            pred_images = center_crop(pred_images, mask.shape[-2:])

            gt_images = apply_mask_and_norm(gt_images, mask, norm_func)
            pred_images = apply_mask_and_norm(pred_images, mask, norm_func)
        else:
            gt_images = norm_func(gt_images)
            pred_images = norm_func(pred_images)

        # Compute psnr and ssim
        for gt, pred in zip(gt_images, pred_images):
            psnr_value = psnr(gt, pred, data_range=gt.max())
            psnr_values.append(psnr_value)

            ssim_value = ssim(gt, pred, data_range=gt.max())*100
            ssim_values.append(ssim_value)

        # Convert list to numpy array
        psnr_values = np.asarray(psnr_values)
        ssim_values = np.asarray(ssim_values)

        # Compute subject reports
        subject_reports = {}
        if subject_ids is not None:
            for i in np.unique(subject_ids):
                idx = np.where(subject_ids == i)[0]
                subject_report = {
                    'psnrs': psnr_values[idx],
                    'ssims': ssim_values[idx],
                    'psnr_mean': np.nanmean(psnr_values[idx]),
                    'ssim_mean': np.nanmean(ssim_values[idx]),
                    'psnr_std': np.nanstd(psnr_values[idx]),
                    'ssim_std': np.nanstd(ssim_values[idx])
                }
                subject_reports[i] = subject_report
            
        # Compute mean and std values
        if subject_ids is not None:
            psnr_mean = np.nanmean([report['psnr_mean'] for report in subject_reports.values()])
            ssim_mean = np.nanmean([report['ssim_mean'] for report in subject_reports.values()])

            psnr_std = np.nanstd([report['psnr_mean'] for report in subject_reports.values()])
            ssim_std = np.nanstd([report['ssim_mean'] for report in subject_reports.values()])
        else:
            psnr_mean = np.nanmean(psnr_values)
            ssim_mean = np.nanmean(ssim_values)

            psnr_std = np.nanstd(psnr_values)
            ssim_std = np.nanstd(ssim_values)
        
        if report_path is not None:
            with open(report_path, 'w') as f:
                f.write(f'PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n')
                f.write(f'SSIM: {ssim_mean:.2f} ± {ssim_std:.2f}\n')
                f.write('\n')

                if subject_ids is not None:
                    for subject_id, report in subject_reports.items():
                        f.write(f'Subject {subject_id}\n')
                        f.write(f'PSNR: {report["psnr_mean"]:.2f} ± {report["psnr_std"]:.2f}\n')
                        f.write(f'SSIM: {report["ssim_mean"]:.2f} ± {report["ssim_std"]:.2f}\n')
                        f.write('\n')         

        res = {
            'psnr_mean': psnr_mean,
            'ssim_mean': ssim_mean,
            'psnr_std': psnr_std,
            'ssim_std': ssim_std,
            'psnrs': psnr_values,
            'ssims': ssim_values,
            'subject_reports': subject_reports
        }

        return res
