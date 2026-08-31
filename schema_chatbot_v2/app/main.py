import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.pipeline_routes import router as pipeline_router
from app.config import settings

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Schema Discovery Chatbot",
    description="Interviews non-technical users to build an IDP target schema and run the Engineer A pipeline.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
