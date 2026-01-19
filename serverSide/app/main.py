from fastapi import FastAPI

print("🚀 Starting FastAPI application...")

try:
    from app.api.pushup_api import router as pushup_router
    print("✅ pushup_api imported successfully")
except Exception as e:
    print("❌ ERROR importing pushup_api:", e)
    raise e

app = FastAPI(title="Pushup AI Backend")

@app.get("/")
def root():
    return {"status": "Pushup API running"}

# ONLY pushup
app.include_router(pushup_router, prefix="/pushup")

print("✅ Pushup router registered")
