from typing import List, Optional
from app.schema.pushup_schema import PushupRepDetail
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PushupPhaseService:
    """Service for tracking pushup phases and counting reps"""
    
    def __init__(self, min_form_score: int = 75):
        self.min_form_score = min_form_score
        self.state = "UP"
        self.pushup_count = 0
        self.last_rep_time = None
        self.rep_details: List[PushupRepDetail] = []
    
    def update_state(self, label: int, form_score: int, timestamp: Optional[float] = None) -> bool:
        """
        Update state machine and count reps
        
        Args:
            label: ML prediction (0=down, 1=up)
            form_score: Form score for current frame
            timestamp: Current timestamp
            
        Returns:
            True if a rep was completed, False otherwise
        """
        rep_completed = False
        
        if self.state == "UP" and label == 0:
            self.state = "DOWN"
        elif self.state == "DOWN" and label == 1:
            if form_score >= self.min_form_score:
                self.pushup_count += 1
                rep_completed = True
                
                rep_duration = None
                if self.last_rep_time and timestamp:
                    rep_duration = timestamp - self.last_rep_time
                
                self.rep_details.append(PushupRepDetail(
                    rep_number=self.pushup_count,
                    form_score=float(form_score),
                    duration=rep_duration,
                    timestamp=timestamp
                ))
                
                if timestamp:
                    self.last_rep_time = timestamp
            
            self.state = "UP"
        
        return rep_completed
    
    def reset(self):
        """Reset the phase tracker"""
        self.state = "UP"
        self.pushup_count = 0
        self.last_rep_time = None
        self.rep_details = []
    
    def get_count(self) -> int:
        """Get current pushup count"""
        return self.pushup_count
    
    def get_rep_details(self) -> List[PushupRepDetail]:
        """Get details of all completed reps"""
        return self.rep_details
