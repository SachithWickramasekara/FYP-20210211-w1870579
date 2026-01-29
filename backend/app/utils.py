"""
Image processing and utility functions for the PixelClear backend.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load and convert image to RGB."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def preprocess_image(img: Image.Image, max_size: Optional[int] = None) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        img: PIL Image
        max_size: Optional max dimension (for large images)
    
    Returns:
        Preprocessed tensor with batch dimension [1, 3, H, W]
    """
    # Resize if too large (optional)
    if max_size:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Convert to tensor and normalize to [0, 1]
    tensor = TF.to_tensor(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor back to PIL Image.
    
    Args:
        tensor: Model output tensor [B, 3, H, W] or [3, H, W]
    
    Returns:
        PIL Image
    """
    # Remove batch dimension if present and clamp to [0, 1]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    return TF.to_pil_image(tensor)


def restore_image(
    model: nn.Module,
    image: Union[Image.Image, Path, str],
    device: torch.device,
    max_size: Optional[int] = None
) -> Image.Image:
    """
    Restore a single image using the trained model.
    
    Args:
        model: Trained restoration model
        image: PIL Image, image path, or image file path
        device: Device to run inference on
        max_size: Optional max dimension (for large images)
    
    Returns:
        Restored PIL Image
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = load_image(image)
    else:
        img = image
    
    # Preprocess
    input_tensor = preprocess_image(img, max_size=max_size).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess
    restored_img = postprocess_image(output_tensor)
    
    return restored_img


def generate_heatmap(
    input_img: Image.Image,
    restored_img: Image.Image,
    colormap: str = 'hot',
    alpha: float = 0.7
) -> Image.Image:
    """
    Generate a heatmap showing pixel-level differences between input and restored images.
    
    Args:
        input_img: Original input image (PIL Image)
        restored_img: Restored image (PIL Image)
        colormap: Matplotlib colormap name (default: 'hot')
        alpha: Transparency for overlay (0-1)
    
    Returns:
        Heatmap as PIL Image
    """
    # Ensure images are the same size
    if input_img.size != restored_img.size:
        restored_img = restored_img.resize(input_img.size, Image.BILINEAR)
    
    # Convert to numpy arrays
    input_arr = np.array(input_img, dtype=np.float32) / 255.0
    restored_arr = np.array(restored_img, dtype=np.float32) / 255.0
    
    # Compute absolute difference
    diff = np.abs(restored_arr - input_arr)
    
    # Convert to grayscale difference
    diff_gray = np.mean(diff, axis=2)
    
    # Normalize to [0, 1]
    if diff_gray.max() > 0:
        diff_gray = diff_gray / diff_gray.max()
    
    # Apply colormap
    try:
        cmap = cm.get_cmap(colormap)
    except AttributeError:
        # For newer matplotlib versions
        cmap = cm.colormaps[colormap]
    heatmap = cmap(diff_gray)
    
    # Convert to uint8
    heatmap_uint8 = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    # Create PIL Image
    heatmap_img = Image.fromarray(heatmap_uint8)
    
    return heatmap_img


def create_comparison_image(
    input_img: Image.Image,
    restored_img: Image.Image,
    labels: Tuple[str, str] = ("Input", "Restored")
) -> Image.Image:
    """
    Create a side-by-side comparison image.
    
    Args:
        input_img: Original input image
        restored_img: Restored image
        labels: Tuple of labels for left and right images
    
    Returns:
        Comparison image as PIL Image
    """
    # Ensure images are the same size
    if input_img.size != restored_img.size:
        restored_img = restored_img.resize(input_img.size, Image.BILINEAR)
    
    # Get dimensions
    width, height = input_img.size
    total_width = width * 2
    total_height = height
    
    # Create comparison image
    comparison = Image.new('RGB', (total_width, total_height))
    comparison.paste(input_img, (0, 0))
    comparison.paste(restored_img, (width, 0))
    
    return comparison


def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded string with data URI prefix
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Return with data URI prefix
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 encoded string to PIL Image.
    
    Args:
        base64_string: Base64 encoded string (with or without data URI prefix)
    
    Returns:
        PIL Image
    """
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_bytes = base64.b64decode(base64_string)
    buffer = io.BytesIO(img_bytes)
    return Image.open(buffer)
