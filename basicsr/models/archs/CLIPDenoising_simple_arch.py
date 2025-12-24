"""
CLIPDenoising - Simplified Architecture (Model-Agnostic)

Simple architecture that works with ANY CLIP model without requiring multi-scale features.

Architecture Flow:
    Input (H×W) → CLIP Encoder (frozen) → Embedding (H/32×W/32 or patch tokens)
                          ↓
                  Learned Deconvolution Decoder
                  (progressive upsampling without skip connections)
                          ↓
                  Upsampled Features (H×W)
                          ↓
              Concatenate [Upsampled Features, Original Input]
                          ↓
                  Convolutional Refinement Layers
                          ↓
                    Output (H×W)

Key Features:
    - Works with ANY CLIP encoder (ViT, ResNet, ConvNeXt, etc.) via timm
    - Uses ONLY the final embedding from CLIP (no multi-scale features needed)
    - No dependency on encoder architecture specifics (unlike original CLIPDenoising)
    - Simple decoder with deconvolution layers for upsampling
    - No skip connections from encoder (unlike U-Net style)
    - Frozen CLIP encoder (transfer learning)
    - Supports both RGB and single-channel (CT) inputs

Requirements:
    pip install timm torch

Usage:
    from basicsr.models.archs.CLIPDenoising_simple_arch import CLIPDenoisingSimple

    # Create model with any CLIP encoder
    model = CLIPDenoisingSimple(
        clip_model_name='resnet50_clip.openai',
        inp_channels=1,  # For CT images
        out_channels=1,
        num_upsample_blocks=5  # For 32x upsampling (2^5 = 32)
    )

    # Forward pass
    clean_image = model(noisy_image)
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class DeconvUpBlock(nn.Module):
    """
    Deconvolution upsampling block (2x upsampling)
    Uses transposed convolution for learned upsampling
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Additional refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.refine(x)
        return x


class SimpleDecoder(nn.Module):
    """
    Simple decoder with deconvolution blocks
    Progressively upsamples from low-resolution CLIP features to full resolution
    NO skip connections from encoder
    """

    def __init__(self, in_channels, num_blocks=5, base_channels=64):
        """
        Args:
            in_channels: Input channel dimension from CLIP encoder
            num_blocks: Number of upsampling blocks (determines upsampling factor = 2^num_blocks)
            base_channels: Base channel count for decoder
        """
        super().__init__()

        # Initial projection: CLIP features -> decoder channels
        self.init_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Deconvolution upsampling blocks
        self.up_blocks = nn.ModuleList()

        # Progressive channel reduction while upsampling
        # Example for 5 blocks: 512 -> 256 -> 128 -> 64 -> 64 -> 64
        channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels, base_channels]

        # Extend if needed
        while len(channels) < num_blocks:
            channels.append(base_channels)

        for i in range(num_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1] if i + 1 < len(channels) else base_channels

            self.up_blocks.append(DeconvUpBlock(in_ch, out_ch))

    def forward(self, x):
        # Initial projection
        x = self.init_proj(x)

        # Progressive upsampling
        for up_block in self.up_blocks:
            x = up_block(x)

        return x


class SimpleConvRefinement(nn.Module):
    """
    Simple convolutional refinement block
    Takes concatenated [upsampled_features, input] and refines to clean output
    """

    def __init__(self, in_channels, out_channels, base_channels=64, num_layers=5):
        super().__init__()

        layers = []

        # First layer: in_channels -> base_channels
        layers.append(nn.Conv2d(in_channels, base_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: base_channels -> base_channels
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(base_channels, base_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        # Final layer: base_channels -> out_channels
        layers.append(nn.Conv2d(base_channels, out_channels, 3, padding=1))

        self.refinement = nn.Sequential(*layers)

    def forward(self, x):
        return self.refinement(x)


class CLIPDenoisingSimple(nn.Module):
    """
    Simplified CLIP Denoising Architecture (Model-Agnostic)

    Architecture:
        Input (H×W) -> CLIP Encoder -> Embedding (H/32×W/32)
                                            ↓
                                  Deconvolution Decoder
                                  (no skip connections)
                                            ↓
                              Upsampled Features (H×W)
                                            ↓
                        Concatenate [Upsampled Features, Input]
                                            ↓
                              Convolutional Refinement
                                            ↓
                                      Output (H×W)

    Args:
        clip_model_name: CLIP model from timm (e.g., 'resnet50_clip.openai', 'vit_base_patch16_clip_224.openai')
        inp_channels: Input channels (1 for CT, 3 for RGB)
        out_channels: Output channels
        num_upsample_blocks: Number of 2x upsampling blocks (e.g., 5 for 32x total upsampling)
        decoder_base_channels: Base channel count for decoder
        refinement_base_channels: Base channel count for refinement layers
        num_refinement_layers: Number of convolutional refinement layers
        freeze_clip: Whether to freeze CLIP encoder
    """

    def __init__(
        self,
        clip_model_name='resnet50_clip.openai',
        inp_channels=3,
        out_channels=3,
        num_upsample_blocks=5,  # 2^5 = 32x upsampling (for typical CLIP ResNet)
        decoder_base_channels=64,
        refinement_base_channels=64,
        num_refinement_layers=5,
        freeze_clip=True,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        self.inp_channels = inp_channels
        self.out_channels = out_channels

        # Initial 1x1 conv for single-channel inputs (e.g., CT images)
        # Converts 1 channel to 3 channels for CLIP encoder
        if inp_channels == 1:
            self.first = nn.Conv2d(inp_channels, 3, kernel_size=1, bias=True)

        # Load CLIP encoder from timm
        print(f"Loading CLIP model: {clip_model_name}")

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
        clip_feature_dim = self.clip_encoder.num_features
        print(f"CLIP feature dimension: {clip_feature_dim}")

        # Simple decoder with deconvolution (NO skip connections)
        self.decoder = SimpleDecoder(
            in_channels=clip_feature_dim,
            num_blocks=num_upsample_blocks,
            base_channels=decoder_base_channels,
        )

        # Simple convolutional refinement
        # Input: [upsampled decoder features + original input]
        # refinement_in_channels = decoder_base_channels + inp_channels
        refinement_in_channels = decoder_base_channels

        self.refinement = SimpleConvRefinement(
            in_channels=refinement_in_channels,
            out_channels=out_channels,
            base_channels=refinement_base_channels,
            num_layers=num_refinement_layers,
        )

    def forward(self, x):
        input_ = x.clone()

        # Convert 1-channel input to 3-channel for CLIP if needed
        if self.inp_channels == 1:
            x_adapted = self.first(x)
        else:
            x_adapted = x

        # Extract features from CLIP encoder (ONLY final embedding)
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

        # Decoder: upsample features to input resolution (NO skip connections)
        upsampled_features = self.decoder(clip_features)

        # Concatenate upsampled features with original input
        # combined = torch.cat([upsampled_features, input_], dim=1)
        combined = upsampled_features

        # Convolutional refinement to output clean image
        output = self.refinement(combined)

        return output

    def train(self, mode=True):
        """Override train to keep CLIP encoder in eval mode if frozen"""
        super().train(mode)
        # Keep CLIP in eval mode if it was frozen
        if hasattr(self, 'clip_encoder') and not any(p.requires_grad for p in self.clip_encoder.parameters()):
            self.clip_encoder.eval()
        return self


# Convenience function to create model variants
def create_clip_denoising_simple(model_type='resnet50', inp_channels=3, out_channels=3, **kwargs):
    """
    Create CLIPDenoisingSimple model with preset configurations

    Args:
        model_type: 'vit_base', 'vit_large', 'resnet50', 'resnet101'
        inp_channels: Input channels
        out_channels: Output channels
        **kwargs: Additional arguments for CLIPDenoisingSimple
    """
    model_configs = {
        'vit_base': {
            'clip_model_name': 'vit_base_patch16_clip_224.openai',
            'num_upsample_blocks': 4,  # ViT patch16 at 224x224 gives 14x14 feature map
        },
        'vit_large': {
            'clip_model_name': 'vit_large_patch14_clip_224.openai',
            'num_upsample_blocks': 4,  # ViT patch14 at 224x224 gives 16x16 feature map
        },
        'resnet50': {
            'clip_model_name': 'resnet50_clip.openai',
            'num_upsample_blocks': 5,  # ResNet typically has 32x downsampling
        },
        'resnet101': {
            'clip_model_name': 'resnet101_clip.openai',
            'num_upsample_blocks': 5,  # ResNet typically has 32x downsampling
        },
    }

    if model_type not in model_configs:
        raise ValueError(f"model_type must be one of {list(model_configs.keys())}")

    config = model_configs[model_type]
    # Override with any user-provided kwargs
    config.update(kwargs)
    config['inp_channels'] = inp_channels
    config['out_channels'] = out_channels

    return CLIPDenoisingSimple(**config)


if __name__ == '__main__':
    # Test the model
    print("=" * 60)
    print("Testing CLIPDenoisingSimple architecture")
    print("=" * 60)

    # Test with RGB input using ResNet
    print("\n[Test 1] RGB input with ResNet-50 CLIP encoder")
    print("-" * 60)
    model = create_clip_denoising_simple(
        model_type='resnet50',
        inp_channels=3,
        out_channels=3,
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

    # Test with single-channel input (CT images)
    print("\n[Test 2] Single-channel input (CT images)")
    print("-" * 60)
    model_ct = create_clip_denoising_simple(
        model_type='resnet50',
        inp_channels=1,
        out_channels=1
    )
    x_ct = torch.randn(2, 1, 512, 512)
    print(f"Input shape: {x_ct.shape}")
    model_ct.eval()
    with torch.no_grad():
        out_ct = model_ct(x_ct)
    print(f"Output shape: {out_ct.shape}")
    assert out_ct.shape == x_ct.shape, "Output shape should match input shape"

    # Test with different resolution
    print("\n[Test 3] Different input resolution")
    print("-" * 60)
    x_large = torch.randn(1, 3, 512, 512)
    print(f"Input shape: {x_large.shape}")
    with torch.no_grad():
        out_large = model(x_large)
    print(f"Output shape: {out_large.shape}")
    assert out_large.shape == x_large.shape, "Should handle different resolutions"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
