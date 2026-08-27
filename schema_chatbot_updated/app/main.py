import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Schema Discovery Chatbot",
    description="Interviews non-technical users to build an IDP target schema.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}
