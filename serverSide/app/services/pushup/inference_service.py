import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_BASE_PATH = Path(__file__).parent.parent.parent.parent / "ML_Model" / "pushup_model" / "saved_models"
DEFAULT_MODEL = "GradientBoosting.pkl"


class PushupInferenceService:
    """Service for pushup inference using ML models"""
    
    def __init__(self, model_name: str = "GradientBoosting"):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the ML model with version compatibility handling"""
        model_path = MODEL_BASE_PATH / f"{self.model_name}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model {self.model_name} not found, using default GradientBoosting")
            model_path = MODEL_BASE_PATH / DEFAULT_MODEL
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            # Try loading with joblib first (better for scikit-learn models)
            try:
                import joblib
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model {self.model_name} using joblib")
            except ImportError:
                # Fallback to pickle
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded model {self.model_name} using pickle")
        except (ValueError, TypeError) as e:
            # Handle scikit-learn version incompatibility
            error_msg = str(e)
            if "incompatible dtype" in error_msg or "missing_go_to_left" in error_msg:
                logger.warning(f"Model version incompatibility detected, trying compatibility mode...")
                try:
                    import joblib
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded model {self.model_name} using joblib (compatibility mode)")
                except:
                    with open(model_path, "rb") as f:
                        try:
                            self.model = pickle.load(f, encoding='latin1')
                        except:
                            self.model = pickle.load(f)
                    logger.info(f"Loaded model {self.model_name} using pickle (compatibility mode)")
            else:
                raise
        
        logger.info(f"Successfully loaded model: {self.model_name}")
    
    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        Predict pushup state (0=down, 1=up) and confidence
        
        Args:
            features: Feature vector from landmarks
            
        Returns:
            tuple: (label, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        label = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0].max()
        
        return int(label), float(proba)
