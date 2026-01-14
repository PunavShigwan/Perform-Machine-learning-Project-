import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PushupFeatureService:
    """Service for extracting features from pose landmarks"""
    
    @staticmethod
    def landmarks_to_features(landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to feature vector
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Feature vector array
        """
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(coords).reshape(1, -1)
    
    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """
        Calculate angle between three points
        
        Args:
            a, b, c: Landmark points
            
        Returns:
            Angle in degrees
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle
    
    @staticmethod
    def calculate_form_score(landmarks) -> int:
        """
        Calculate form score based on shoulder-hip-ankle alignment
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Form score (0-100)
        """
        shoulder = landmarks[11]  # left shoulder
        hip = landmarks[23]       # left hip
        ankle = landmarks[27]     # left ankle
        angle = PushupFeatureService.calculate_angle(shoulder, hip, ankle)
        score = max(0, 100 - abs(180 - angle) * 2)
        return int(score)
    
    @staticmethod
    def calculate_elbow_angle(landmarks) -> float:
        """
        Calculate elbow angle to determine up/down position
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Elbow angle in degrees
        """
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]
        return PushupFeatureService.calculate_angle(shoulder, elbow, wrist)
