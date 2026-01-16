from pydantic import BaseModel

class PushupAnalysisResponse(BaseModel):
    pushup_count: int
    average_form: int
    estimated_max_range: str
    output_video_path: str
