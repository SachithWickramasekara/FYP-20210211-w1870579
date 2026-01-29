"""
Configuration settings for the PixelClear backend.
"""

import os
from pathlib import Path
import torch

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Model configuration
MODEL_CONFIG = {
    "base_channels": 32,
    "num_blocks": 4,
    "dropout": 0.0,
}

# Model checkpoint path
CHECKPOINT_PATH = BASE_DIR / "experiments" / "baseline_training" / "best_model.pth"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File upload settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp"
}

# Directory paths
UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
RESULTS_DIR = BASE_DIR / "backend" / "results"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Image processing settings
MAX_IMAGE_SIZE = 2048  # Maximum dimension for large images (optional resizing)

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

# API settings
API_PREFIX = "/api"
