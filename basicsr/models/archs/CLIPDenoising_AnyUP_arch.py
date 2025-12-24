"""
CLIPDenoising with AnyUP architecture
Uses single-scale CLIP features + AnyUP upsampling + simple U-Net decoder

Architecture Flow:
    Input (H×W) → CLIP Encoder (frozen) → Features (H/32×W/32)
                          ↓
    Input (H×W) → Downsample to H/2×W/2 (guidance image)
                          ↓
              AnyUp(guidance, features) → Upsampled Features (H/2×W/2)
                          ↓
                  Feature Projection (channel adjustment)
                          ↓
            Simple Decoder (refine + upsample 2x)
                          ↓
                    Output (H×W)

Key Features:
    - Works with ANY CLIP encoder (ViT, ResNet, ConvNeXt, etc.) via timm
    - Uses AnyUp for intelligent feature upsampling guided by input image
    - Simpler architecture compared to multi-scale skip connections
    - Frozen CLIP encoder (transfer learning)
    - Supports both RGB and single-channel (CT) inputs

Requirements:
    pip install timm torch
    # AnyUp will be loaded automatically from torch.hub on first use

Usage:
    from basicsr.models.archs.CLIPDenoising_AnyUP_arch import create_clip_denoising_anyup

    # Create model with ViT-Base CLIP encoder
    model = create_clip_denoising_anyup(model_type='vit_base', inp_channels=3, out_channels=3)

    # Or with ResNet-50 CLIP encoder
    model = create_clip_denoising_anyup(model_type='resnet50', inp_channels=3, out_channels=3)

    # Forward pass
    clean_image = model(noisy_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class AnyUPWrapper(nn.Module):
    """
    Wrapper for AnyUp upsampler from torch.hub

    AnyUp takes:
        - hr_image: high-resolution guidance image (B, 3, H, W)
        - lr_features: low-resolution features to upsample (B, C, h, w)

    Returns:
        - hr_features: upsampled features at HR resolution (B, C, H, W)

    Reference: https://wimmerth.github.io/anyup/
    """

    def __init__(self):
        super().__init__()
        self.upsampler = torch.hub.load("wimmerth/anyup", "anyup")

        for param in self.upsampler.parameters():
            param.requires_grad = False
        self.upsampler.eval()

    def forward(self, hr_image, lr_features):
        """
        Args:
            hr_image: High-resolution input image (B, 3, H, W)
            lr_features: Low-resolution features from encoder (B, C, h, w)

        Returns:
            hr_features: Upsampled features (B, C, H, W)
        """
        return self.upsampler(hr_image, lr_features)


class SimpleDecoder(nn.Module):
    """
    Simple decoder WITHOUT skip connections
    Takes features from AnyUp (H/2×W/2) and refines/upsamples to output (H×W)

    Uses learned deconvolution (transposed convolution) for upsampling
    """

    def __init__(self, in_channels, out_channels=3, base_channels=64):
        super().__init__()

        # Initial feature processing
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Learned upsampling block: H/2 → H (2x) using transposed convolution
        self.deconv_up = nn.ConvTranspose2d(
            base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.deconv_bn = nn.BatchNorm2d(base_channels)
        self.deconv_relu = nn.ReLU(inplace=True)

        # Refinement after upsampling
        self.refine1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final refinement and output
        self.final_refine = nn.Sequential(
            nn.Conv2d(base_channels + out_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(base_channels // 2, out_channels, 3, padding=1)

    def forward(self, x, input_):
        # Initial processing at H/2×W/2
        x = self.init_conv(x)

        # Learned deconvolution upsampling to H×W
        x = self.deconv_up(x)
        x = self.deconv_bn(x)
        x = self.deconv_relu(x)

        # Refine upsampled features
        x = self.refine1(x)

        # Final refinement
        x = torch.cat([x, input_], dim=1)
        x = self.final_refine(x)
        out = self.output(x)

        return out


class CLIPDenoisingAnyUP(nn.Module):
    """
    CLIP + AnyUP + Simple Decoder Architecture

    Architecture:
        Input (H×W) -> CLIP Encoder -> Embedding (H/32×W/32)
                                            ↓
        Input (H×W) -> Downsample (H/2×W/2) [guidance]
                                            ↓
                  AnyUp(guidance, embedding) -> Features (H/2×W/2)
                                            ↓
                                    Feature Projection
                                            ↓
                          Simple Decoder (refine + 2x upsample)
                                            ↓
                                      Output (H×W)

    Args:
        clip_model_name: CLIP model from timm (e.g., 'vit_base_patch16_clip_224.openai')
        inp_channels: Input channels (1 for CT, 3 for RGB)
        out_channels: Output channels
        decoder_base_channels: Base channel count for decoder
        freeze_clip: Whether to freeze CLIP encoder
        anyup_target_size: Target resolution after AnyUP (e.g., 'half' for H/2×W/2)
    """

    def __init__(
        self,
        clip_model_name="vit_base_patch16_clip_224.openai",
        inp_channels=3,
        out_channels=3,
        decoder_base_channels=64,
        freeze_clip=True,
        anyup_target_resolution="half",  # 'half' for H/2×W/2, 'full' for H×W
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        self.inp_channels = inp_channels
        self.anyup_target_resolution = anyup_target_resolution

        # Initial 1x1 conv for single-channel inputs (e.g., CT images)
        # Converts 1 channel to 3 channels for CLIP encoder
        if inp_channels == 1:
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=True)
            inp_channels = 3

        # Load CLIP encoder from timm
        print(f"Loading CLIP model: {clip_model_name}")

        # Check if using ViT model (has fixed input size)
        is_vit = "vit" in clip_model_name.lower()

        if is_vit:
            print("WARNING: ViT models have fixed input size (typically 224x224)")
            print("For flexible input sizes, use ResNet-based CLIP models:")
            print("  - model_type='resnet50' or model_type='resnet101'")

        self.clip_encoder = timm.create_model(
            clip_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling to get spatial features
            dynamic_img_size=True,  # Allow dynamic input sizes
        )

        # Freeze CLIP encoder if specified
        if freeze_clip:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
            self.clip_encoder.eval()
            print("CLIP encoder frozen")

        # Get the feature dimension from CLIP
        # This depends on the model architecture
        clip_feature_dim = self.clip_encoder.num_features
        print(f"CLIP feature dimension: {clip_feature_dim}")

        # AnyUP upsampling module from torch.hub
        # Takes hr_image and lr_features, outputs upsampled features
        self.anyup = AnyUPWrapper()
        self.anyup.eval()

        # Channel projection after AnyUp (CLIP features -> decoder input channels)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(clip_feature_dim, decoder_base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Simple decoder without skip connections
        # Input is H/2×W/2, output is H×W
        self.decoder = SimpleDecoder(
            in_channels=decoder_base_channels * 2,
            out_channels=out_channels,
            base_channels=decoder_base_channels,
        )

    def forward(self, x):
        input_ = x.clone()

        # Convert 1-channel input to 3-channel for CLIP if needed
        if self.inp_channels == 1:
            # Use learned 1x1 conv to convert 1ch -> 3ch
            x_adapted = self.first(x)
            # For AnyUp guidance, replicate the original 1-channel to 3 channels
            hr_image = x.repeat(1, 3, 1, 1)
        else:
            x_adapted = x
            hr_image = x

        # Extract features from CLIP encoder
        # For ViT models, this gives patch embeddings
        # For ResNet models, this gives final conv features
        with torch.set_grad_enabled(not self.training or self.clip_encoder.training):
            clip_features = self.clip_encoder(x_adapted)

        # Handle different output formats from CLIP models
        if isinstance(clip_features, (list, tuple)):
            # Some models return multiple outputs, take the last one
            clip_features = clip_features[-1]

        # If features are flattened (ViT), reshape to spatial
        if clip_features.dim() == 3:  # [B, N, C]
            B, N, C = clip_features.shape
            # Assume square spatial dimensions
            H = W = int((N - 1) ** 0.5)  # -1 for class token if present
            if (H * W + 1) == N:  # Has class token
                clip_features = clip_features[:, 1:, :]  # Remove class token
            clip_features = clip_features.transpose(1, 2).reshape(B, C, H, W)

        # Prepare HR image for AnyUp based on target resolution
        if self.anyup_target_resolution == "half":
            # Downsample to H/2×W/2 for AnyUp
            hr_image_for_anyup = F.interpolate(
                hr_image, scale_factor=0.5, mode="bilinear", align_corners=False
            )
        else:  # 'full'
            hr_image_for_anyup = hr_image

        # AnyUP upsampling: uses hr_image to guide upsampling of lr_features
        # lr_features: (B, C, h, w) -> hr_features: (B, C, H, W)
        upsampled_features = self.anyup(hr_image_for_anyup, clip_features)

        # Project features to decoder input channels
        upsampled_features = self.feature_proj(upsampled_features)

        # Simple decoder: refines features to output clean image
        output = self.decoder(upsampled_features, input_)

        return output

    def train(self, mode=True):
        """Override train to keep CLIP encoder in eval mode if frozen"""
        super().train(mode)
        # Keep CLIP in eval mode if it was frozen
        if hasattr(self, "clip_encoder") and not any(
            p.requires_grad for p in self.clip_encoder.parameters()
        ):
            self.clip_encoder.eval()
        return self


# Convenience function to create model variants
def create_clip_denoising_anyup(
    model_type="vit_base", inp_channels=3, out_channels=3, **kwargs
):
    """
    Create CLIPDenoisingAnyUP model with preset configurations

    Args:
        model_type: 'vit_base', 'vit_large', 'resnet50', 'resnet101'
        inp_channels: Input channels
        out_channels: Output channels
        **kwargs: Additional arguments for CLIPDenoisingAnyUP
    """
    model_configs = {
        "vit_base": "vit_base_patch16_clip_224.openai",
        "vit_large": "vit_large_patch14_clip_224.openai",
        "resnet50": "resnet50_clip.openai",
        "resnet101": "resnet101_clip.openai",
    }

    if model_type not in model_configs:
        raise ValueError(f"model_type must be one of {list(model_configs.keys())}")

    return CLIPDenoisingAnyUP(
        clip_model_name=model_configs[model_type],
        inp_channels=inp_channels,
        out_channels=out_channels,
        **kwargs,
    )


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing CLIPDenoisingAnyUP architecture")
    print("=" * 60)

    # Test with RGB input using ResNet (supports variable input sizes)
    print("\n[Test 1] RGB input with ResNet-50 CLIP encoder")
    print("-" * 60)
    model = create_clip_denoising_anyup(
        model_type="resnet50",  # ResNet supports variable input sizes
        inp_channels=3,
        out_channels=3,
        anyup_target_resolution="half",
    )
    x = torch.randn(2, 3, 256, 256)

    print(f"Input shape: {x.shape}")
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters (CLIP): {total_params - trainable_params:,}")

    # Test with different resolution
    print("\n[Test 2] Different input resolution")
    print("-" * 60)
    x_large = torch.randn(1, 3, 512, 512)
    print(f"Input shape: {x_large.shape}")
    with torch.no_grad():
        out_large = model(x_large)
    print(f"Output shape: {out_large.shape}")
    assert out_large.shape == x_large.shape, "Should handle different resolutions"

    # Test with single-channel input (CT images)
    print("\n[Test 3] Single-channel input (CT images)")
    print("-" * 60)
    model_ct = create_clip_denoising_anyup(
        model_type="resnet50", inp_channels=1, out_channels=1
    )
    x_ct = torch.randn(2, 1, 256, 256)
    print(f"Input shape: {x_ct.shape}")
    model_ct.eval()
    with torch.no_grad():
        out_ct = model_ct(x_ct)
    print(f"Output shape: {out_ct.shape}")
    assert out_ct.shape == x_ct.shape, "Output shape should match input shape"
