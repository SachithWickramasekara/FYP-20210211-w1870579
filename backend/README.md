# PixelClear Backend API

FastAPI backend service for PixelClear image restoration with explainability heatmaps.

## Features

- Image restoration using trained PyTorch model
- Explainability heatmap generation
- Side-by-side comparison images
- RESTful API with JSON responses
- Base64 encoded images for easy frontend integration
- File download endpoints

## Setup

### Prerequisites

- Python 3.8+
- Trained model checkpoint at `experiments/baseline_training/best_model.pth`

### Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The backend uses configuration from `app/config.py`. Key settings:

- **Model checkpoint**: `experiments/baseline_training/best_model.pth`
- **Max file size**: 10MB
- **Allowed formats**: PNG, JPG, JPEG, WEBP
- **Max image size**: 2048px (optional resizing for large images)

## Running the Server

### Development Mode

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

**GET** `/health`

Check server and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "checkpoint_path": "/path/to/best_model.pth"
}
```

### Image Restoration

**POST** `/api/restore`

Upload an image and get restored image, heatmap, and comparison.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "status": "success",
  "restored_image": "data:image/png;base64,iVBORw0KG...",
  "heatmap": "data:image/png;base64,iVBORw0KG...",
  "comparison": "data:image/png;base64,iVBORw0KG...",
  "processing_time": 1.234,
  "session_id": "uuid-string"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/restore" \
  -F "file=@path/to/image.jpg"
```

### Download Endpoints

**GET** `/api/download/restored/{session_id}`

Download the restored image.

**GET** `/api/download/heatmap/{session_id}`

Download the heatmap image.

**GET** `/api/download/comparison/{session_id}`

Download the comparison image.

### Compare Images

**POST** `/api/compare`

Generate a side-by-side comparison of two images.

**Request:**
- Content-Type: `multipart/form-data`
- Body: 
  - `input_file` (original image)
  - `restored_file` (restored image)

**Response:**
```json
{
  "status": "success",
  "comparison": "data:image/png;base64,iVBORw0KG..."
}
```

## Error Responses

The API returns standard HTTP status codes:

- **200 OK**: Success
- **413 Payload Too Large**: File size exceeds 10MB
- **415 Unsupported Media Type**: Invalid file type
- **422 Unprocessable Entity**: Image processing error
- **503 Service Unavailable**: Model not loaded

Error response format:
```json
{
  "detail": "Error message description"
}
```

## File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py        # PyTorch model architecture
│   ├── utils.py         # Image processing utilities
│   └── config.py        # Configuration settings
├── uploads/             # Temporary uploads (auto-created)
├── results/              # Processed results (auto-created)
├── requirements.txt
└── README.md
```

## CORS Configuration

The backend is configured to accept requests from common frontend development ports:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)
- `http://localhost:8080` (Vue default)

To add additional origins, edit `app/config.py`.

## Model Information

- **Architecture**: BaselineRestorationModel (NAFNet-inspired)
- **Parameters**: ~16K (0.02M)
- **Base channels**: 32
- **Number of blocks**: 4
- **Input/Output**: RGB images, any size (resized if > 2048px)

## Troubleshooting

### Model Not Loading

Ensure the checkpoint exists at:
```
experiments/baseline_training/best_model.pth
```

### CUDA Not Available

The backend will automatically use CPU if CUDA is not available. Check the `/health` endpoint to see the device being used.

### Large Images

Images larger than 2048px in any dimension will be automatically resized. You can override this by passing `max_size` parameter in the restore endpoint.

## Development

### Testing the API

Use the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Frontend Integration

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/restore', {
  method: 'POST',
  body: formData
});

const data = await response.json();
// data.restored_image contains base64 encoded image
// data.heatmap contains base64 encoded heatmap
// data.comparison contains base64 encoded comparison
```

## License

Part of the PixelClear Final Year Project.
