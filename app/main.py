from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import classify, describe, detect, remove_bg, analyze


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered visual inspection API for e-commerce product images.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/analyze", tags=["Analysis"])
app.include_router(describe.router, prefix="/analyze", tags=["Analysis"])
app.include_router(detect.router, prefix="/analyze", tags=["Analysis"])
app.include_router(remove_bg.router, prefix="/analyze", tags=["Analysis"])
app.include_router(analyze.router, prefix="/analyze", tags=["Unified"])


@app.on_event("startup")
async def startup_event():
    """Load all models at startup so first request is not slow."""
    print("[Startup] Pre-loading all models...")
    from app.services.classifier import load_model as load_classifier
    from app.services.descriptor import load_model as load_descriptor
    from app.services.detector import load_model as load_detector
    from app.services.bg_remover import load_model as load_bg_remover
    load_classifier()
    load_descriptor()
    load_detector()
    load_bg_remover()
    print("[Startup] All models loaded and ready.")


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
