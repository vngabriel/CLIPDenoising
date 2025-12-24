import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.append("/home/gabriel/Research/CLIPDenoising")

from basicsr.models.archs.CLIPDenoising_simple_arch import CLIPDenoisingSimple
from basicsr.metrics.CT_psnr_ssim import compute_PSNR, compute_SSIM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================================
# USER CONFIGURATION - Change these paths to your image files
# ============================================================================

# Path to input noisy image (LDCT)
INPUT_IMAGE_PATH = "/home/gabriel/Research/dataset/mayo-challenge/npy_img/1mm B30/L506_250_input.npy"

# Path to ground truth clean image
GROUND_TRUTH_PATH = INPUT_IMAGE_PATH.replace("_input.npy", "_target.npy")

# HU windowing for display (adjust based on tissue type)
# Brain: (-20, 120), Soft tissue: (-50, 150), Abdomen: (-160, 240), Bone: (-500, 2000)
DISPLAY_MIN = -160  # HU
DISPLAY_MAX = 240   # HU

# ============================================================================

# CT data normalization parameters
TRUNC_MIN = -1024
TRUNC_MAX = 3072

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
        psnr: PSNR value
        ssim: SSIM value
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

        # Compute metrics
        psnr = compute_PSNR(
            restored,
            img_clean,
            data_range=TRUNC_MAX - TRUNC_MIN,
            trunc_min=TRUNC_MIN,
            trunc_max=TRUNC_MAX,
            norm_range_max=3096,
            norm_range_min=-1024,
        )

        ssim = compute_SSIM(
            restored,
            img_clean,
            data_range=TRUNC_MAX - TRUNC_MIN,
            trunc_min=TRUNC_MIN,
            trunc_max=TRUNC_MAX,
            norm_range_max=3096,
            norm_range_min=-1024,
        )

        # Convert to numpy for visualization
        input_np = input_tensor.squeeze().cpu().numpy()
        restored_np = restored.squeeze().cpu().numpy()
        gt_np = img_clean.squeeze().cpu().numpy()

    return input_np, restored_np, gt_np, psnr, ssim


def plot_comparison(input_img, restored_img, gt_img, psnr, ssim,
                   display_min=DISPLAY_MIN, display_max=DISPLAY_MAX):
    """
    Plot input, restored, and ground truth images side by side.

    Args:
        input_img: Input noisy image
        restored_img: Denoised output
        gt_img: Ground truth clean image
        psnr: PSNR value
        ssim: SSIM value
        display_min: Minimum HU value for windowing
        display_max: Maximum HU value for windowing
    """
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    # Input image
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(input_img, cmap='gray', vmin=display_min, vmax=display_max)
    ax1.set_title('Input (LDCT)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='HU')

    # Restored image
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(restored_img, cmap='gray', vmin=display_min, vmax=display_max)
    ax2.set_title(f'Predicted (Denoised)\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='HU')

    # Ground truth
    ax3 = plt.subplot(gs[2])
    im3 = ax3.imshow(gt_img, cmap='gray', vmin=display_min, vmax=display_max)
    ax3.set_title('Ground Truth (NDCT)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='HU')

    plt.suptitle('CT Image Denoising Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save figure
    output_filename = 'ct_denoising_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_filename}")

    plt.show()


def main():
    """Main function to run the visualization."""
    print("=" * 80)
    print("CT Image Denoising Visualization")
    print("=" * 80)
    print(f"Input image: {INPUT_IMAGE_PATH}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    print(f"Display window: [{DISPLAY_MIN}, {DISPLAY_MAX}] HU")
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
    input_img, restored_img, gt_img, psnr, ssim = process_image(
        model, INPUT_IMAGE_PATH, GROUND_TRUTH_PATH
    )

    print("\nResults:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.3f}")
    print(f"  Image shape: {input_img.shape}")

    # Plot comparison
    print("\nGenerating visualization...")
    plot_comparison(input_img, restored_img, gt_img, psnr, ssim)

    print("\nDone!")


if __name__ == "__main__":
    main()
