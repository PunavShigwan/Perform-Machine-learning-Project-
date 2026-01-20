from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

print("🚀 Starting FastAPI application...")

# =====================================================
# CREATE APP
# =====================================================
app = FastAPI(title="Pushup AI Backend")

# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # DEV ONLY
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ROUTES (PREFIX ONLY HERE)
# =====================================================
try:
    from app.api.pushup_api import router as pushup_router
    print("✅ pushup_api imported successfully")
except Exception as e:
    print("❌ ERROR importing pushup_api:", e)
    raise e

# ✅ PREFIX APPLIED HERE (matches your pushup_api.py)
app.include_router(pushup_router, prefix="/pushup")

# =====================================================
# STATIC FILES (PROCESSED VIDEO DOWNLOAD)
# =====================================================
app.mount(
    "/app/uploads",
    StaticFiles(directory="app/uploads"),
    name="uploads"
)

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status": "Pushup API running"}

print("✅ Pushup router registered")
