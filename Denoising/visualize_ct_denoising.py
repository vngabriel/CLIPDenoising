import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import gridspec

sys.path.append("/home/gabriel/Research/CLIPDenoising")

from basicsr.metrics.CT_psnr_ssim import compute_PSNR, compute_SSIM, denormalize_
from basicsr.models.archs.CLIPDenoising_simple_arch import CLIPDenoisingSimple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================================
# USER CONFIGURATION - Change these paths to your image files
# ============================================================================

# Path to input noisy image (LDCT)
INPUT_IMAGE_PATH = (
    "/home/gabriel/Research/dataset/mayo-challenge/npy_img/1mm B30/L506_250_input.npy"
)

# Path to ground truth clean image
GROUND_TRUTH_PATH = INPUT_IMAGE_PATH.replace("_input.npy", "_target.npy")

# HU windowing for display (adjust based on tissue type)
# Brain: (-20, 120), Soft tissue: (-50, 150), Abdomen: (-160, 240), Bone: (-500, 2000)
# Set USE_AUTO_WINDOW=True to automatically adjust window based on data range
DISPLAY_MIN = -160  # HU
DISPLAY_MAX = 240  # HU
USE_AUTO_WINDOW = True  # Recommended when image contains bones

# ============================================================================

# CT data normalization parameters (matching test_real_denoising_CT.py)
NORM_RANGE_MIN = -1024
NORM_RANGE_MAX = 3096


def load_model():
    """Load the pretrained CLIPDenoising model."""
    model_restoration = CLIPDenoisingSimple(
        clip_model_name="vit_base_patch16_clip_224.openai",
        inp_channels=1,
        out_channels=1,
        num_upsample_blocks=4,
    )

    checkpoint_path = "./experiments_refine/CLIPDenoisingSimple_LDCTDenoising_GaussianSigma5_archived_20251214_220357/models/net_g_latest.pth"
    checkpoint = torch.load(checkpoint_path)
    model_restoration.load_state_dict(checkpoint["params"])

    model_restoration.cuda()
    model_restoration.eval()

    return model_restoration


def process_image(model, input_path, gt_path):
    """
    Process an image through the denoising model.

    Args:
        model: The loaded denoising model
        input_path: Path to the noisy input image (.npy)
        gt_path: Path to the ground truth clean image (.npy)

    Returns:
        input_img: Input noisy image (numpy array)
        restored_img: Denoised output (numpy array)
        gt_img: Ground truth clean image (numpy array)
        psnr_input: PSNR value for input vs ground truth
        ssim_input: SSIM value for input vs ground truth
        psnr_restored: PSNR value for restored vs ground truth
        ssim_restored: SSIM value for restored vs ground truth
    """
    factor = 32

    with torch.no_grad():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # Load ground truth
        img_clean = np.load(gt_path)[..., np.newaxis]
        img_clean = torch.from_numpy(img_clean).permute(2, 0, 1)
        img_clean = img_clean.unsqueeze(0).float().cuda()

        # Load noisy input
        img = np.load(input_path)[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_tensor = img.unsqueeze(0).float().cuda()

        # Padding to ensure dimensions are multiples of 32
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_padded = F.pad(input_tensor, (0, padw, 0, padh), "reflect")

        # Run inference
        restored = model(input_padded)

        # Unpad to original dimensions
        restored = restored[:, :, :h, :w]

        # Compute metrics for input (noisy) image
        psnr_input = compute_PSNR(
            input_tensor,
            img_clean,
            data_range=NORM_RANGE_MAX - NORM_RANGE_MIN,
            trunc_min=DISPLAY_MIN,
            trunc_max=DISPLAY_MAX,
            norm_range_max=NORM_RANGE_MAX,
            norm_range_min=NORM_RANGE_MIN,
        )

        ssim_input = compute_SSIM(
            input_tensor,
            img_clean,
            data_range=NORM_RANGE_MAX - NORM_RANGE_MIN,
            trunc_min=DISPLAY_MIN,
            trunc_max=DISPLAY_MAX,
            norm_range_max=NORM_RANGE_MAX,
            norm_range_min=NORM_RANGE_MIN,
        )

        # Compute metrics for restored (denoised) image
        psnr_restored = compute_PSNR(
            restored,
            img_clean,
            data_range=NORM_RANGE_MAX - NORM_RANGE_MIN,
            trunc_min=DISPLAY_MIN,
            trunc_max=DISPLAY_MAX,
            norm_range_max=NORM_RANGE_MAX,
            norm_range_min=NORM_RANGE_MIN,
        )

        ssim_restored = compute_SSIM(
            restored,
            img_clean,
            data_range=NORM_RANGE_MAX - NORM_RANGE_MIN,
            trunc_min=DISPLAY_MIN,
            trunc_max=DISPLAY_MAX,
            norm_range_max=NORM_RANGE_MAX,
            norm_range_min=NORM_RANGE_MIN,
        )

        # Convert to numpy for visualization
        input_np = input_tensor.squeeze().cpu().numpy()
        restored_np = restored.squeeze().cpu().numpy()
        gt_np = img_clean.squeeze().cpu().numpy()

    return (
        input_np,
        restored_np,
        gt_np,
        psnr_input,
        ssim_input,
        psnr_restored,
        ssim_restored,
    )


def plot_comparison(
    input_img,
    restored_img,
    gt_img,
    psnr_input,
    ssim_input,
    psnr_restored,
    ssim_restored,
    display_min=DISPLAY_MIN,
    display_max=DISPLAY_MAX,
):
    """
    Plot input, restored, and ground truth images side by side.

    Args:
        input_img: Input noisy image (normalized)
        restored_img: Denoised output (normalized)
        gt_img: Ground truth clean image (normalized)
        psnr_input: PSNR value for input vs ground truth
        ssim_input: SSIM value for input vs ground truth
        psnr_restored: PSNR value for restored vs ground truth
        ssim_restored: SSIM value for restored vs ground truth
        display_min: Minimum HU value for windowing
        display_max: Maximum HU value for windowing
    """
    # Denormalize images to HU values using existing function
    input_hu = denormalize_(
        input_img, norm_range_max=NORM_RANGE_MAX, norm_range_min=NORM_RANGE_MIN
    )
    restored_hu = denormalize_(
        restored_img, norm_range_max=NORM_RANGE_MAX, norm_range_min=NORM_RANGE_MIN
    )
    gt_hu = denormalize_(
        gt_img, norm_range_max=NORM_RANGE_MAX, norm_range_min=NORM_RANGE_MIN
    )

    print("\nDenormalized HU ranges:")
    print(f"  Input - min: {input_hu.min():.1f}, max: {input_hu.max():.1f} HU")
    print(f"  Restored - min: {restored_hu.min():.1f}, max: {restored_hu.max():.1f} HU")
    print(f"  GT - min: {gt_hu.min():.1f}, max: {gt_hu.max():.1f} HU")

    # Determine display window
    if USE_AUTO_WINDOW:
        # Auto-compute window from actual data range
        vmin = min(input_hu.min(), restored_hu.min(), gt_hu.min())
        vmax = max(input_hu.max(), restored_hu.max(), gt_hu.max())
        print(f"\nUsing auto window: [{vmin:.1f}, {vmax:.1f}] HU")
    else:
        # Use fixed window specified by user
        vmin = display_min
        vmax = display_max
        print(f"\nUsing fixed window: [{vmin:.1f}, {vmax:.1f}] HU")

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    # Input image
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(input_hu, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title(
        f"Input (LDCT)\nPSNR: {psnr_input:.2f} dB | SSIM: {ssim_input:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="HU")

    # Restored image
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(restored_hu, cmap="gray", vmin=vmin, vmax=vmax)
    psnr_improvement = psnr_restored - psnr_input
    ssim_improvement = ssim_restored - ssim_input
    ax2.set_title(
        f"Denoised\nPSNR: {psnr_restored:.2f} dB (+{psnr_improvement:.2f}) | SSIM: {ssim_restored:.3f} (+{ssim_improvement:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="HU")

    # Ground truth
    ax3 = plt.subplot(gs[2])
    im3 = ax3.imshow(gt_hu, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("Ground Truth (NDCT)", fontsize=14, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="HU")

    plt.suptitle(
        "CT Image Denoising Comparison", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    # Save figure
    output_filename = "ct_denoising_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_filename}")

    plt.show()


def main():
    """Main function to run the visualization."""
    print("=" * 80)
    print("CT Image Denoising Visualization")
    print("=" * 80)
    print(f"Input image: {INPUT_IMAGE_PATH}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    if USE_AUTO_WINDOW:
        print("Display window: Auto (based on data range)")
    else:
        print(f"Display window: [{DISPLAY_MIN}, {DISPLAY_MAX}] HU (fixed)")
    print("=" * 80)

    # Check if files exist
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at {INPUT_IMAGE_PATH}")
        return

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: Ground truth image not found at {GROUND_TRUTH_PATH}")
        return

    # Load model
    print("\nLoading model...")
    model = load_model()
    print("Model loaded successfully!")

    # Process image
    print("\nProcessing image through denoising model...")
    (
        input_img,
        restored_img,
        gt_img,
        psnr_input,
        ssim_input,
        psnr_restored,
        ssim_restored,
    ) = process_image(model, INPUT_IMAGE_PATH, GROUND_TRUTH_PATH)

    print("\nMetrics (vs Ground Truth):")
    print(f"  Input  - PSNR: {psnr_input:.2f} dB | SSIM: {ssim_input:.3f}")
    print(f"  Denoised - PSNR: {psnr_restored:.2f} dB | SSIM: {ssim_restored:.3f}")
    print(
        f"  Improvement - PSNR: +{psnr_restored - psnr_input:.2f} dB | SSIM: +{ssim_restored - ssim_input:.3f}"
    )
    print(f"\nImage shape: {input_img.shape}")
    print("\nData ranges:")
    print(f"  Input - min: {input_img.min():.2f}, max: {input_img.max():.2f}")
    print(f"  Restored - min: {restored_img.min():.2f}, max: {restored_img.max():.2f}")
    print(f"  GT - min: {gt_img.min():.2f}, max: {gt_img.max():.2f}")

    # Plot comparison
    print("\nGenerating visualization...")
    plot_comparison(
        input_img,
        restored_img,
        gt_img,
        psnr_input,
        ssim_input,
        psnr_restored,
        ssim_restored,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
