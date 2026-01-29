"""
FastAPI application for PixelClear image restoration service.
"""

import time
import uuid
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.models import BaselineRestorationModel
from app.config import (
    MODEL_CONFIG,
    CHECKPOINT_PATH,
    DEVICE,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    UPLOADS_DIR,
    RESULTS_DIR,
    MAX_IMAGE_SIZE,
    CORS_ORIGINS,
    API_PREFIX,
)
from app.utils import (
    load_image,
    restore_image,
    generate_heatmap,
    create_comparison_image,
    image_to_base64,
)

# Initialize FastAPI app
app = FastAPI(
    title="PixelClear API",
    description="Image restoration API with explainability heatmaps",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[BaselineRestorationModel] = None
model_loaded = False


def load_model():
    """Load the trained model on application startup."""
    global model, model_loaded
    
    try:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {CHECKPOINT_PATH}. "
                "Please ensure the model has been trained and the checkpoint exists."
            )
        
        print(f"Loading model from {CHECKPOINT_PATH}")
        
        # Create model instance
        model = BaselineRestorationModel(
            base_channels=MODEL_CONFIG["base_channels"],
            num_blocks=MODEL_CONFIG["num_blocks"],
            dropout=MODEL_CONFIG["dropout"]
        ).to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Device: {DEVICE}")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best PSNR: {checkpoint.get('best_val_psnr', checkpoint.get('val_psnr', 'unknown')):.2f} dB")
        
        model_loaded = True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_loaded = False
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
        "checkpoint_path": str(CHECKPOINT_PATH),
    }


@app.post(f"{API_PREFIX}/restore")
async def restore(
    file: UploadFile = File(...),
    max_size: Optional[int] = None
):
    """
    Upload an image and restore it using the trained model.
    
    Returns:
        JSON response with restored image, heatmap, and comparison as base64 strings
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"MIME type not supported: {file.content_type}"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Validate file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Load image from bytes
        input_image = Image.open(io.BytesIO(file_content))
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Use provided max_size or default from config
        processing_max_size = max_size if max_size else MAX_IMAGE_SIZE
        
        # Record start time
        start_time = time.time()
        
        # Restore image
        restored_image = restore_image(
            model=model,
            image=input_image,
            device=DEVICE,
            max_size=processing_max_size
        )
        
        # Generate heatmap
        heatmap_image = generate_heatmap(input_image, restored_image)
        
        # Create comparison image
        comparison_image = create_comparison_image(input_image, restored_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save images to results directory (optional, for download endpoints)
        results_path = RESULTS_DIR / session_id
        results_path.mkdir(exist_ok=True)
        
        input_image.save(results_path / "input.png")
        restored_image.save(results_path / "restored.png")
        heatmap_image.save(results_path / "heatmap.png")
        comparison_image.save(results_path / "comparison.png")
        
        # Convert to base64
        restored_base64 = image_to_base64(restored_image)
        heatmap_base64 = image_to_base64(heatmap_image)
        comparison_base64 = image_to_base64(comparison_image)
        
        return JSONResponse({
            "status": "success",
            "restored_image": restored_base64,
            "heatmap": heatmap_base64,
            "comparison": comparison_base64,
            "processing_time": round(processing_time, 3),
            "session_id": session_id,
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing image: {str(e)}"
        )


@app.get(f"{API_PREFIX}/download/restored/{{session_id}}")
async def download_restored(session_id: str):
    """Download the restored image for a given session."""
    file_path = RESULTS_DIR / session_id / "restored.png"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restored image not found for session {session_id}"
        )
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"restored_{session_id}.png"
    )


@app.get(f"{API_PREFIX}/download/heatmap/{{session_id}}")
async def download_heatmap(session_id: str):
    """Download the heatmap image for a given session."""
    file_path = RESULTS_DIR / session_id / "heatmap.png"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Heatmap not found for session {session_id}"
        )
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"heatmap_{session_id}.png"
    )


@app.get(f"{API_PREFIX}/download/comparison/{{session_id}}")
async def download_comparison(session_id: str):
    """Download the comparison image for a given session."""
    file_path = RESULTS_DIR / session_id / "comparison.png"
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison image not found for session {session_id}"
        )
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=f"comparison_{session_id}.png"
    )


@app.post(f"{API_PREFIX}/compare")
async def compare_images(
    input_file: UploadFile = File(...),
    restored_file: UploadFile = File(...)
):
    """
    Generate a side-by-side comparison of two images.
    
    Args:
        input_file: Original input image
        restored_file: Restored image
    
    Returns:
        Comparison image as base64 string
    """
    try:
        # Load images
        input_content = await input_file.read()
        restored_content = await restored_file.read()
        
        input_image = Image.open(io.BytesIO(input_content))
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        restored_image = Image.open(io.BytesIO(restored_content))
        if restored_image.mode != 'RGB':
            restored_image = restored_image.convert('RGB')
        
        # Create comparison
        comparison_image = create_comparison_image(input_image, restored_image)
        comparison_base64 = image_to_base64(comparison_image)
        
        return JSONResponse({
            "status": "success",
            "comparison": comparison_base64,
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error creating comparison: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
