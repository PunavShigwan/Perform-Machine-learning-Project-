import os
import cv2
import time
import numpy as np
import mediapipe as mp
import pickle
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
from pathlib import Path

from app.schema.pushup_schema import PushupAnalysisRequest, PushupAnalysisResponse, PushupRepDetail
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/pushup", tags=["pushup"])

# ===== MediaPipe setup =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== Model paths =====
MODEL_BASE_PATH = Path(__file__).parent.parent.parent / "ML_Model" / "pushup_model" / "saved_models"
DEFAULT_MODEL = "GradientBoosting.pkl"


def load_model(model_name: str = "GradientBoosting"):
    """Load the ML model for pushup classification with version compatibility handling"""
    model_path = MODEL_BASE_PATH / f"{model_name}.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model {model_name} not found, using default GradientBoosting")
        model_path = MODEL_BASE_PATH / DEFAULT_MODEL
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Try loading with joblib first (better for scikit-learn models)
        try:
            import joblib
            model = joblib.load(model_path)
            logger.info(f"Loaded model {model_name} using joblib")
        except ImportError:
            # Fallback to pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model {model_name} using pickle")
    except (ValueError, TypeError) as e:
        # Handle scikit-learn version incompatibility
        error_msg = str(e)
        if "incompatible dtype" in error_msg or "missing_go_to_left" in error_msg:
            logger.error(f"Model version incompatibility detected: {e}")
            logger.info("Attempting to fix with scikit-learn compatibility workaround...")
            
            # Try loading with a workaround for version compatibility
            try:
                import sklearn
                import warnings
                warnings.filterwarnings('ignore')
                
                # Try loading with joblib with compatibility mode
                try:
                    import joblib
                    model = joblib.load(model_path)
                    logger.info(f"Loaded model {model_name} using joblib (compatibility mode)")
                except:
                    # Last resort: try pickle with encoding
                    with open(model_path, "rb") as f:
                        # Try different pickle protocols
                        try:
                            model = pickle.load(f, encoding='latin1')
                        except:
                            model = pickle.load(f)
                    logger.info(f"Loaded model {model_name} using pickle (compatibility mode)")
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model loading failed due to version incompatibility. "
                           f"Please ensure scikit-learn >= 1.3.0 is installed. "
                           f"Error: {str(e)}"
                )
        else:
            raise
    
    logger.info(f"Successfully loaded model: {model_name}")
    return model


def extract_landmarks(image):
    """Extract pose landmarks from image using MediaPipe"""
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks.landmark, results
    return None, None


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def form_score(landmarks):
    """Calculate form score based on shoulder-hip-ankle alignment"""
    shoulder = landmarks[11]  # left shoulder
    hip = landmarks[23]       # left hip
    ankle = landmarks[27]     # left ankle
    angle = calculate_angle(shoulder, hip, ankle)
    score = max(0, 100 - abs(180 - angle) * 2)
    return int(score)


def elbow_angle(landmarks):
    """Calculate elbow angle to determine up/down position"""
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]
    return calculate_angle(shoulder, elbow, wrist)


def landmarks_to_features(landmarks):
    """Convert landmarks to feature vector for ML model"""
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(coords).reshape(1, -1)


def update_prediction(last_form, last_duration, current_pred):
    """Update predicted max pushups based on form and duration"""
    if last_duration is None:
        return current_pred
    
    form_factor = (last_form - 70) / 30  # >70 good, <70 penalize
    speed_factor = 1.5 if last_duration < 1 else (0.8 if last_duration > 3 else 1.0)
    
    adjustment = int(form_factor * 2 * speed_factor)
    new_pred = max(5, current_pred + adjustment)
    return new_pred


def analyze_pushup_video(video_path: str, model, min_form_score: int = 75):
    """
    Analyze pushup video and return count, form scores, and estimated target
    
    Args:
        video_path: Path to video file
        model: Loaded ML model
        min_form_score: Minimum form score to count a rep
    
    Returns:
        dict with pushup_count, estimated_target, rep_details, average_form_score
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    pushup_count = 0
    state = "UP"
    last_rep_time = None
    predicted_max = 15  # initial baseline
    form_scores = []
    rep_details = []
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            landmarks, results = extract_landmarks(frame)
            
            if landmarks:
                features = landmarks_to_features(landmarks)
                score = form_score(landmarks)
                form_scores.append(score)
                
                elbow = elbow_angle(landmarks)
                elbow_label = 0 if elbow < 90 else 1
                
                # Predict using ML model
                label = model.predict(features)[0]  # 0=down, 1=up
                proba = model.predict_proba(features)[0].max()
                
                # Fallback to elbow angle if confidence is low
                if proba < 0.6:
                    label = elbow_label
                
                # State machine for rep counting
                if state == "UP" and label == 0:
                    state = "DOWN"
                elif state == "DOWN" and label == 1:
                    if score >= min_form_score:
                        pushup_count += 1
                        now = time.time()
                        rep_duration = None
                        if last_rep_time:
                            rep_duration = now - last_rep_time
                        
                        rep_details.append(PushupRepDetail(
                            rep_number=pushup_count,
                            form_score=float(score),
                            duration=rep_duration,
                            timestamp=now
                        ))
                        
                        # Update prediction
                        predicted_max = update_prediction(score, rep_duration, predicted_max)
                        last_rep_time = now
                    
                    state = "UP"
    
    finally:
        cap.release()
    
    average_form_score = np.mean(form_scores) if form_scores else 0.0
    
    return {
        "pushup_count": pushup_count,
        "estimated_target": int(predicted_max),
        "rep_details": rep_details,
        "average_form_score": float(average_form_score)
    }


@router.post("/analyze", response_model=PushupAnalysisResponse)
async def analyze_pushup(
    video: UploadFile = File(..., description="Video file to analyze"),
    min_form_score: Optional[int] = Form(75, description="Minimum form score to count a rep"),
    model_name: Optional[str] = Form("GradientBoosting", description="ML model to use")
):
    """
    Analyze pushup video and return count, form analysis, and estimated target
    
    - **video**: Video file (mp4, avi, mov)
    - **min_form_score**: Minimum form score (0-100) to count a rep (default: 75)
    - **model_name**: ML model name (default: GradientBoosting)
    
    Returns:
    - Pushup count
    - Estimated target
    - Form scores per rep
    - Average form score
    """
    start_time = time.time()
    
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temp file
        suffix = Path(video.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Load model
        try:
            model = load_model(model_name)
        except FileNotFoundError as e:
            logger.error(f"Model loading error: {e}")
            raise HTTPException(status_code=500, detail=f"Model not found: {str(e)}")
        
        # Analyze video
        try:
            results = analyze_pushup_video(temp_path, model, min_form_score)
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
        
        processing_time = time.time() - start_time
        
        response = PushupAnalysisResponse(
            status="success",
            filename=video.filename,
            pushups=results["pushup_count"],
            estimated_target=results["estimated_target"],
            average_form_score=results["average_form_score"],
            rep_details=results["rep_details"],
            processing_time=processing_time,
            message=f"Successfully analyzed {results['pushup_count']} pushups"
        )
        
        logger.info(f"Analysis complete: {results['pushup_count']} pushups, target: {results['estimated_target']}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pushup_api"}
