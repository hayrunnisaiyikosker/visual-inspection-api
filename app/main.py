from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import classify, describe, detect, remove_bg, analyze
from app.api.routes.auth import router as auth_router
from app.models.database import create_tables

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

app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(classify.router, prefix="/analyze", tags=["Analysis"])
app.include_router(describe.router, prefix="/analyze", tags=["Analysis"])
app.include_router(detect.router, prefix="/analyze", tags=["Analysis"])
app.include_router(remove_bg.router, prefix="/analyze", tags=["Analysis"])
app.include_router(analyze.router, prefix="/analyze", tags=["Unified"])


@app.on_event("startup")
async def startup_event():
    print("[Startup] Creating database tables...")
    await create_tables()
    print("[Startup] Models will load on first request (lazy loading).")


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


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

static_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/ui", tags=["Frontend"])
async def frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))
