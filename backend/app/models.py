"""
PyTorch Model Architecture for Image Restoration.

This module contains the BaselineRestorationModel architecture used for training
and inference. The model is based on NAFNet-inspired architecture.
"""

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Simple gating mechanism for activation."""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block: Simple and effective restoration block."""
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.act = SimpleGate()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        return x + residual


class BaselineRestorationModel(nn.Module):
    """
    Lightweight baseline restoration model.
    
    Architecture:
    - Input conv: 3 -> base_channels
    - N NAFBlocks for feature extraction
    - Output conv: base_channels -> 3
    - Residual connection from input
    """
    
    def __init__(self, base_channels: int = 32, num_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        
        self.input_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        
        # Stack of NAF blocks
        self.blocks = nn.Sequential(*[
            NAFBlock(base_channels, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.output_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input degraded image [B, 3, H, W]
            
        Returns:
            Restored image [B, 3, H, W]
        """
        residual = x
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)
        return x + residual  # Residual connection
