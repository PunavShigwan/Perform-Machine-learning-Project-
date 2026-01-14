# Pushup API Testing Guide

## Quick Start

### 1. Install Dependencies

```bash
cd serverSide
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

From the `serverSide` directory:

```bash
# Option 1: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python -m app.main
```

The server will start at: `http://localhost:8000`

### 3. Test the API

#### Option A: Using the Test Script

```bash
# Run the automated test script
python test_pushup_api.py

# Or with a specific video
python test_pushup_api.py path/to/your/video.mp4
```

#### Option B: Using Browser

1. Open `http://localhost:8000` - Should show API info
2. Open `http://localhost:8000/docs` - Interactive API documentation (Swagger UI)
3. Open `http://localhost:8000/redoc` - Alternative API documentation

#### Option C: Using curl

```bash
# Test health endpoint
curl http://localhost:8000/pushup/health

# Test root endpoint
curl http://localhost:8000/

# Test pushup analysis (replace with your video path)
curl -X POST "http://localhost:8000/pushup/analyze" \
  -F "video=@path/to/your/video.mp4" \
  -F "min_form_score=75" \
  -F "model_name=GradientBoosting"
```

#### Option D: Using Python requests

```python
import requests

# Upload and analyze video
with open('your_video.mp4', 'rb') as video:
    files = {'video': video}
    data = {'min_form_score': 75, 'model_name': 'GradientBoosting'}
    response = requests.post('http://localhost:8000/pushup/analyze', files=files, data=data)
    print(response.json())
```

## API Endpoints

### 1. Health Check
- **GET** `/health` - Global health check
- **GET** `/pushup/health` - Pushup API health check

### 2. Root
- **GET** `/` - API information

### 3. Pushup Analysis
- **POST** `/pushup/analyze`
  - **Parameters:**
    - `video` (file): Video file to analyze (mp4, avi, mov, mkv)
    - `min_form_score` (int, optional): Minimum form score (default: 75)
    - `model_name` (string, optional): ML model name (default: "GradientBoosting")
  
  - **Response:**
    ```json
    {
      "status": "success",
      "filename": "video.mp4",
      "pushups": 10,
      "estimated_target": 15,
      "average_form_score": 85.5,
      "rep_details": [...],
      "processing_time": 12.34
    }
    ```

## Interactive API Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test the API directly from these pages by uploading a video file.

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Model not found error
- Ensure the model file exists at: `serverSide/ML_Model/pushup_model/saved_models/GradientBoosting.pkl`
- Check the path in `pushup_api.py` if models are in a different location

### Video processing fails
- Verify video format is supported (mp4, avi, mov, mkv)
- Check video file is not corrupted
- Ensure MediaPipe can process the video (person visible, good lighting)

### Import errors
- Make sure you're running from the correct directory
- Check that `app` module is in Python path
- Try: `python -m app.main` instead of `python app/main.py`

## Expected Output

When testing successfully, you should see:

```
✅ Analysis Successful!
Filename: test_video.mp4
Pushups Counted: 10
Estimated Target: 15
Average Form Score: 85.50%
Processing Time: 12.34 seconds
```
