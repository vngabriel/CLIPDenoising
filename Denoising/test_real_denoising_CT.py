import argparse
import os
import sys

import numpy as np

from basicsr.models.archs.CLIPDenoising_simple_arch import CLIPDenoisingSimple

sys.path.append("/home/gabriel/Research/CLIPDenoising")
from glob import glob

import torch
import torch.nn.functional as F
import utils
from pylinac.core.nps import (
    average_power,
    noise_power_spectrum_1d,
    noise_power_spectrum_2d,
)
from tqdm import tqdm

from basicsr.metrics.CT_psnr_ssim import compute_PSNR, compute_SSIM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Synthetic Color Denoising")

parser.add_argument(
    "--input_dir",
    default="/home/gabriel/Research/dataset/mayo-challenge/npy_img",
    type=str,
    help="Directory of validation images",
)

args = parser.parse_args()


def proc(tar_img, prd_img):
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM


def compute_NPS_metrics(img_array, pixel_size=1.0, roi_size=64):
    """
    Compute Noise Power Spectrum (NPS) metrics using pylinac.

    NPS quantifies noise characteristics in the frequency domain.
    Lower NPS values indicate less noise.

    Args:
        img_array: 2D numpy array of the image
        pixel_size: Pixel size in mm (default 1.0)
        roi_size: Size of ROI for NPS computation (default 64)

    Returns:
        nps_2d: 2D NPS array
        nps_1d: 1D radial average of NPS
        avg_power: Average power of the NPS (summary metric)
    """
    h, w = img_array.shape

    # Extract multiple ROIs from the image
    num_rois_h = h // roi_size
    num_rois_w = w // roi_size

    rois = []
    for i in range(num_rois_h):
        for j in range(num_rois_w):
            roi = img_array[
                i * roi_size : (i + 1) * roi_size, j * roi_size : (j + 1) * roi_size
            ]
            rois.append(roi)

    # Compute 2D NPS using pylinac
    nps_2d = noise_power_spectrum_2d(pixel_size=pixel_size, rois=rois)

    # Compute 1D NPS (radial average)
    nps_1d = noise_power_spectrum_1d(spectrum_2d=nps_2d)

    # Compute average power
    avg_power = average_power(nps_1d)

    return nps_2d, nps_1d, avg_power


# network arch
"""
type: CLIPDenoising
inp_channels: 1
out_channels: 1
depth: 5
wf: 64
num_blocks: [3, 4, 6, 3]
bias: false
model_path: /data0/cj/model_data/ldm/stable-diffusion/RN50.pt

aug_level: 0.025
"""

# model_restoration = CLIPDenoising(
#     inp_channels=1,
#     out_channels=1,
#     depth=5,
#     wf=64,
#     num_blocks=[3, 4, 6, 3],
#     bias=False,
#     model_path="/home/gabriel/Research/CLIPDenoising/RN50.pt",
#     aug_level=0.025,
# )
# checkpoint = torch.load("./Denoising/pretrained_models/LDCT/net_g_latest.pth")
# checkpoint = torch.load(
#     "./experiments_refine/CLIPDenoising_LDCTDenoising_GaussianSigma5_archived_20251217_221309/models/net_g_latest.pth"
# )
# model_restoration = CLIPDenoisingAnyUP(
#     clip_model_name="vit_base_patch16_clip_224.openai",
#     inp_channels=1,
#     out_channels=1,
# )
# checkpoint = torch.load(
#     "./experiments_refine/CLIPDenoisingAnyUP_LDCTDenoising_GaussianSigma5/models/net_g_latest.pth"
# )
model_restoration = CLIPDenoisingSimple(
    # clip_model_name="resnet50_clip.openai",
    clip_model_name="vit_base_patch16_clip_224.openai",
    inp_channels=1,
    out_channels=1,
    num_upsample_blocks=4,
)
checkpoint = torch.load(
    "./experiments_refine/CLIPDenoisingSimple_LDCTDenoising_GaussianSigma5/models/net_g_latest.pth"
)
# model_restoration = Unet(
#     in_channels=1,
#     out_channels=1,
#     channel=64,
#     output_activation=None,
#     use_batchnorm=True,
# )
# checkpoint = torch.load(
#     "./experiments_refine/Unet_LDCTDenoising_GaussianSigma5/models/net_g_latest.pth"
# )
load_result = model_restoration.load_state_dict(checkpoint["params"])

model_restoration.cuda()
model_restoration.eval()
##########################

factor = 32

test_patient = "L506"
# test_patient = ["L096", "L506"]
target_path = sorted(glob(os.path.join(args.input_dir, "*/*target*")))
input_path = sorted(glob(os.path.join(args.input_dir, "*/*input*")))

# input_ = []
# target_ = []
# for inp, targ in zip(input_path, target_path):
#     for pat in test_patient:
#         if pat in inp:
#             input_.append(inp)
#         if pat in targ:
#             target_.append(targ)

input_ = [f for f in input_path if test_patient in f and "1mm" in f and "B30" in f]
target_ = [f for f in target_path if test_patient in f and "1mm" in f and "B30" in f]

# input_ = [f for f in input_path if test_patient in f and "3mm" in f and "D45" in f]
# target_ = [f for f in target_path if test_patient in f and "3mm" in f and "D45" in f]

# input_ = [f for f in input_path if test_patient in f]
# target_ = [f for f in target_path if test_patient in f]


trunc_min = -1024
trunc_max = 3072
shape_ = 512
psnr_list = []
ssim_list = []
nps_input_list = []
nps_output_list = []
nps_clean_list = []

for noise, clean in tqdm(zip(input_, target_)):
    with torch.no_grad():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_clean = np.load(clean)[..., np.newaxis]
        img_clean = torch.from_numpy(img_clean).permute(2, 0, 1)
        img_clean = img_clean.unsqueeze(0).float().cuda()

        img = np.load(noise)[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).float().cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        psnr, ssim = (
            compute_PSNR(
                restored,
                img_clean,
                data_range=trunc_max - trunc_min,
                trunc_min=trunc_min,
                trunc_max=trunc_max,
                norm_range_max=3096,
                norm_range_min=-1024,
            ),
            compute_SSIM(
                restored,
                img_clean,
                data_range=trunc_max - trunc_min,
                trunc_min=trunc_min,
                trunc_max=trunc_max,
                norm_range_max=3096,
                norm_range_min=-1024,
            ),
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        # Compute NPS for input, output, and clean images
        # Convert to numpy arrays for NPS computation
        input_np = input_[:, :, :h, :w].squeeze().cpu().numpy()
        restored_np = restored.squeeze().cpu().numpy()
        clean_np = img_clean.squeeze().cpu().numpy()

        # Compute NPS using pylinac (pixel_size=1.0 for normalized frequency)
        _, _, avg_power_input = compute_NPS_metrics(
            input_np, pixel_size=1.0, roi_size=64
        )
        _, _, avg_power_output = compute_NPS_metrics(
            restored_np, pixel_size=1.0, roi_size=64
        )
        _, _, avg_power_clean = compute_NPS_metrics(
            clean_np, pixel_size=1.0, roi_size=64
        )

        # Store average power values
        nps_input_list.append(avg_power_input)
        nps_output_list.append(avg_power_output)
        nps_clean_list.append(avg_power_clean)


print("\n" + "=" * 80)
print("CT Dataset Evaluation Results")
print("=" * 80)
print(
    "PSNR: {:.2f} dB | SSIM: {:.3f}".format(
        sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list)
    )
)
print("\nNoise Power Spectrum (NPS) - Average Power:")
print("  Input (noisy):     {:.6f}".format(sum(nps_input_list) / len(nps_input_list)))
print("  Output (denoised): {:.6f}".format(sum(nps_output_list) / len(nps_output_list)))
print("  Clean (reference): {:.6f}".format(sum(nps_clean_list) / len(nps_clean_list)))
print(
    "  NPS Reduction:     {:.2f}%".format(
        (
            1
            - (sum(nps_output_list) / len(nps_output_list))
            / (sum(nps_input_list) / len(nps_input_list))
        )
        * 100
    )
)

print("\n" + "-" * 80)
print("NPS Interpretation Guide:")
print("  - NPS measures noise power in the frequency domain")
print("  - Lower values = less noise")
print("  - Ideal denoising: Output NPS should approach Clean (reference) NPS")
print("  - A small reduction (~2%) may indicate:")
print("    1. The model is preserving image structure (good!)")
print("    2. There's room for improvement in noise reduction")
print("    3. The clean reference may already have some residual noise")
print("-" * 80)

print("\nNote: MTF computation requires phantom images with known line pair")
print("      patterns and is not computed for general patient CT images.")

print("=" * 80)
