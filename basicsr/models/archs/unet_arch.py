"""
UNet Architecture for Image Denoising

Classic U-Net architecture for image-to-image translation tasks including denoising.
Implements encoder-decoder structure with skip connections.

Original paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
Reference: arXiv:1505.04597

Usage:
    from basicsr.models.archs.unet_arch import Unet

    # For CT denoising (1-channel)
    model = Unet(in_channels=1, out_channels=1, channel=64)

    # For RGB denoising (3-channel)
    model = Unet(in_channels=3, out_channels=3, channel=64)

    # Forward pass
    clean_image = model(noisy_image)
"""

from typing import Optional

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Activation

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        act_fn: Activation function ('relu' or 'leaky_relu')
        use_batchnorm: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: str = "relu",
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm

        # When using batch norm, conv layers don't need bias
        use_bias = not use_batchnorm

        # First convolution
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
        )

        # Second convolution
        self.conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
        )

        # Batch normalization (optional)
        if use_batchnorm:
            self.batchnorm0 = nn.BatchNorm2d(out_channels)
            self.batchnorm1 = nn.BatchNorm2d(out_channels)

        # Activation function
        act_fn = act_fn.lower()
        if act_fn == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_fn in ["lrelu", "leaky_relu"]:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        # First conv block
        x = self.conv0(x)
        if self.use_batchnorm:
            x = self.batchnorm0(x)
        x = self.activation(x)

        # Second conv block
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = self.activation(x)

        return x


class Unet(nn.Module):
    """
    U-Net architecture for image-to-image translation.

    Original paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
    Reference: arXiv:1505.04597

    Architecture:
        Encoder (Downsampling):
            Input → [64] → MaxPool → [128] → MaxPool → [256] → MaxPool → [512] → MaxPool

        Bottleneck:
            [1024]

        Decoder (Upsampling with Skip Connections):
            Deconv → Concat[512] → [512] → Deconv → Concat[256] → [256] →
            Deconv → Concat[128] → [128] → Deconv → Concat[64] → [64] → Output

    Args:
        in_channels: Number of input channels (1 for grayscale CT images, 3 for RGB)
        out_channels: Number of output channels (1 for CT denoising, 3 for RGB)
        channel: Base number of channels (will be scaled: 64, 128, 256, 512, 1024)
        output_activation: Output activation function ('sigmoid', 'tanh', or None)
        use_batchnorm: Whether to use batch normalization

    Example:
        >>> model = Unet(in_channels=1, out_channels=1, channel=64)
        >>> x = torch.randn(4, 1, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([4, 1, 512, 512])
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channel: int = 64,
        output_activation: Optional[str] = None,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        # Calculate channel scaling factor
        div = 64 / channel

        # Encoder (downsampling path)
        self.block0_conv = DoubleConv(
            in_channels, int(64 // div), act_fn="relu", use_batchnorm=use_batchnorm
        )

        self.block1_conv = DoubleConv(
            int(64 // div), int(128 // div), act_fn="relu", use_batchnorm=use_batchnorm
        )

        self.block2_conv = DoubleConv(
            int(128 // div), int(256 // div), act_fn="relu", use_batchnorm=use_batchnorm
        )

        self.block3_conv = DoubleConv(
            int(256 // div), int(512 // div), act_fn="relu", use_batchnorm=use_batchnorm
        )

        # Bottleneck
        self.block4_conv = DoubleConv(
            int(512 // div),
            int(1024 // div),
            act_fn="relu",
            use_batchnorm=use_batchnorm,
        )
        self.block4_deconv = nn.ConvTranspose2d(
            int(1024 // div), int(512 // div), kernel_size=2, stride=2, padding=0
        )

        # Decoder (upsampling path)
        self.block5_conv = DoubleConv(
            int(1024 // div),  # Concatenated: 512 + 512
            int(512 // div),
            act_fn="relu",
            use_batchnorm=use_batchnorm,
        )
        self.block5_deconv = nn.ConvTranspose2d(
            int(512 // div), int(256 // div), kernel_size=2, stride=2, padding=0
        )

        self.block6_conv = DoubleConv(
            int(512 // div),  # Concatenated: 256 + 256
            int(256 // div),
            act_fn="relu",
            use_batchnorm=use_batchnorm,
        )
        self.block6_deconv = nn.ConvTranspose2d(
            int(256 // div), int(128 // div), kernel_size=2, stride=2, padding=0
        )

        self.block7_conv = DoubleConv(
            int(256 // div),  # Concatenated: 128 + 128
            int(128 // div),
            act_fn="relu",
            use_batchnorm=use_batchnorm,
        )
        self.block7_deconv = nn.ConvTranspose2d(
            int(128 // div), int(64 // div), kernel_size=2, stride=2, padding=0
        )

        self.block8_conv = DoubleConv(
            int(128 // div),  # Concatenated: 64 + 64
            int(64 // div),
            act_fn="relu",
            use_batchnorm=use_batchnorm,
        )

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(
            int(64 // div), out_channels, kernel_size=1, padding=0
        )

        # Max pooling for encoder
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output activation
        self.output_activation = (
            output_activation.lower() if output_activation else None
        )
        if self.output_activation not in [None, "sigmoid", "tanh"]:
            raise ValueError(f"Invalid output_activation: {output_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Encoder path with skip connections
        # BLOCK 0
        x0 = self.block0_conv(x)
        x = self.maxpool(x0)

        # BLOCK 1
        x1 = self.block1_conv(x)
        x = self.maxpool(x1)

        # BLOCK 2
        x2 = self.block2_conv(x)
        x = self.maxpool(x2)

        # BLOCK 3
        x3 = self.block3_conv(x)
        x = self.maxpool(x3)

        # Bottleneck - BLOCK 4
        x = self.block4_conv(x)
        x = self.block4_deconv(x)

        # Decoder path with skip connections
        # BLOCK 5
        x = torch.cat([x, x3], dim=1)
        x = self.block5_conv(x)
        x = self.block5_deconv(x)

        # BLOCK 6
        x = torch.cat([x, x2], dim=1)
        x = self.block6_conv(x)
        x = self.block6_deconv(x)

        # BLOCK 7
        x = torch.cat([x, x1], dim=1)
        x = self.block7_conv(x)
        x = self.block7_deconv(x)

        # BLOCK 8
        x = torch.cat([x, x0], dim=1)
        x = self.block8_conv(x)

        # Final 1x1 convolution
        x = self.final_conv(x)

        # Apply output activation
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)

        return x

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing UNet architecture")
    print("=" * 60)

    # Test with single-channel input (CT images)
    print("\n[Test 1] Single-channel input (CT images)")
    print("-" * 60)
    model = Unet(in_channels=1, out_channels=1, channel=64)
    x = torch.randn(2, 1, 256, 256)
    print(f"Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"

    # Count parameters
    total_params = model.count_parameters()
    print(f"Total trainable parameters: {total_params:,}")

    # Test with RGB input
    print("\n[Test 2] RGB input (3-channel)")
    print("-" * 60)
    model_rgb = Unet(in_channels=3, out_channels=3, channel=64)
    x_rgb = torch.randn(2, 3, 256, 256)
    print(f"Input shape: {x_rgb.shape}")

    model_rgb.eval()
    with torch.no_grad():
        out_rgb = model_rgb(x_rgb)
    print(f"Output shape: {out_rgb.shape}")
    assert out_rgb.shape == x_rgb.shape, "Output shape should match input shape"

    # Test with different resolution
    print("\n[Test 3] Different input resolution")
    print("-" * 60)
    x_large = torch.randn(1, 1, 512, 512)
    print(f"Input shape: {x_large.shape}")
    with torch.no_grad():
        out_large = model(x_large)
    print(f"Output shape: {out_large.shape}")
    assert out_large.shape == x_large.shape, "Should handle different resolutions"

    # Test with batch normalization
    print("\n[Test 4] With batch normalization")
    print("-" * 60)
    model_bn = Unet(in_channels=1, out_channels=1, channel=64, use_batchnorm=True)
    print(f"Model with BatchNorm created successfully")
    with torch.no_grad():
        out_bn = model_bn(x)
    print(f"Output shape: {out_bn.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
