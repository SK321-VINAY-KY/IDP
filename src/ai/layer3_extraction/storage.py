"""
File: storage.py
Purpose: Single-table PostgreSQL schema for IDP extraction results.

  Table: extraction_runs
    One row per extraction request.
    page_details JSONB column stores all per-page engine/confidence info
    keyed by page number (as string) — no separate page_results table needed.

    page_details shape:
    {
        "1": {"engines_used": ["paddleocr_printed"], "confidence": 0.99,
              "escalated": false, "low_confidence": false,
              "capabilities": ["has_digital_text"]},
        "2": {"engines_used": ["vlm_transcribe"], "confidence": 0.71,
              "escalated": true, "low_confidence": true,
              "capabilities": ["has_handwriting"]}
    }

Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-30 (page_details JSONB column)
"""
from datetime import datetime, UTC

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config.settings import settings

Base = declarative_base()
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    doc_id                  = Column(String, nullable=False)   # PDF filename
    page_count              = Column(Integer)                  # total pages in document
    schema_name             = Column(String, nullable=False)   # field keys used for this run
    result_json             = Column(JSONB, nullable=False)    # extracted field values
    page_details            = Column(JSONB)                    # per-page engine/confidence info
    llm_provider            = Column(String)                   # "sarvam" | "ollama"
    model_name              = Column(String)                   # "sarvam-105b" | "qwen2.5:7b"
    processing_time_seconds = Column(Float)
    created_at              = Column(DateTime, default=lambda: datetime.now(UTC))


def init_db():
    Base.metadata.create_all(engine)


def save_extraction_run(
    doc_id: str,
    page_count: int | None,
    schema_name: str,
    result_json: dict,
    llm_provider: str | None,
    model_name: str | None,
    processing_time_seconds: float | None,
    page_outputs: list | None = None,
) -> int:
    """
    Save one extraction run.

    Args:
        page_outputs: list of PageOutput objects from Layer 1+2.
                      If provided, per-page detail is packed into page_details JSONB.
                      If None (e.g. fixture run), page_details is left null.

    Returns:
        The auto-generated run id.
    """
    page_details = None
    if page_outputs:
        page_details = {
            str(p.page_number): {
                "engines_used":   list(getattr(p, "engines_used", []) or []),
                "confidence":     getattr(p, "confidence", None),
                "escalated":      getattr(p, "escalated", None),
                "low_confidence": getattr(p, "low_confidence", None),
                "capabilities":   list(getattr(p, "capabilities", []) or []),
            }
            for p in page_outputs
        }

    session = SessionLocal()
    try:
        run = ExtractionRun(
            doc_id=doc_id,
            page_count=page_count,
            schema_name=schema_name,
            result_json=result_json,
            page_details=page_details,
            llm_provider=llm_provider,
            model_name=model_name,
            processing_time_seconds=processing_time_seconds,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id
    finally:
        session.close()
