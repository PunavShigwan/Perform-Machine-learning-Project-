from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.pushup_api import router as pushup_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fitness ML API",
    description="API for analyzing fitness exercises using machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pushup_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fitness ML API",
        "version": "1.0.0",
        "endpoints": {
            "pushup": "/pushup/analyze",
            "health": "/pushup/health"
        }
    }


@app.get("/health")
async def health():
    """Global health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
