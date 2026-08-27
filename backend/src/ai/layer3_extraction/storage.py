"""
File: storage.py
Purpose: Single shared table capturing Layer 1/2 routing/conversion detail
         (per-document aggregated) and Layer 3 extraction results.
Owner: engineer-b@idp-pilot
Created: 2026-08-20 | Updated: 2026-08-25 (expanded fields per Layer 1/2 schemas)
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, UTC

from src.config.settings import settings

Base = declarative_base()
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


class ProcessingResult(Base):
    __tablename__ = "processing_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    doc_id = Column(String, nullable=False)
    page_count = Column(Integer)

    routes_used = Column(JSONB)
    engines_used = Column(JSONB)
    primary_scripts = Column(JSONB)
    avg_confidence = Column(Float)
    min_confidence = Column(Float)
    low_confidence_page_count = Column(Integer)
    escalated_page_count = Column(Integer)
    total_escalation_attempts = Column(Integer)
    max_complexity_score = Column(Integer)
    has_images = Column(Boolean)

    schema_name = Column(String, nullable=False)
    result_json = Column(JSONB, nullable=False)
    llm_provider = Column(String)
    model_name = Column(String)
    strategy_used = Column(String)
    processing_time_seconds = Column(Float)

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


def init_db():
    Base.metadata.create_all(engine)


def save_processing_result(**kwargs) -> int:
    session = SessionLocal()
    try:
        row = ProcessingResult(**kwargs)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id
    finally:
        session.close()