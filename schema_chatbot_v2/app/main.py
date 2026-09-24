import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.api.pipeline_routes import router as pipeline_router
from app.api.auth_routes import router as auth_router
from app.api.user_routes import router as user_router
from app.api.admin_routes import router as admin_router
from app.config import settings
import app.core.log_buffer  # Attach BufferHandler to root logger

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Schema Discovery & Extraction Pipeline",
    description="Interviews non-technical users to build an IDP target schema and run the extraction pipeline.",
    version="0.2.0",
)

import os

def get_cors_origins() -> list[str]:
    app_env = os.getenv("APP_ENV", "development").lower()
    raw = os.getenv("CORS_ORIGINS")
    if app_env in ("production", "staging"):
        if not raw or not raw.strip():
            raise ValueError(f"CORS_ORIGINS environment variable must be set in {app_env} environment.")
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        if "*" in origins:
            raise ValueError("Wildcard '*' is strictly forbidden in CORS_ORIGINS for production/staging environments.")
        return origins
    else:
        if raw and raw.strip():
            return [o.strip() for o in raw.split(",") if o.strip()]
        return ["http://localhost:8000", "http://127.0.0.1:8000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/app") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(user_router)
app.include_router(router)
app.include_router(pipeline_router)

STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app")


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}
