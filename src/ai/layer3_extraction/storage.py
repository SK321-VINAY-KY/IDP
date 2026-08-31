"""
File: storage.py
Purpose: PostgreSQL storage schema for IDP System.

  Tables:
    1. documents:
       Stores uploaded input documents (e.g. PDF binaries, size, hash, metadata).
    2. schemas:
       Stores confirmed target schemas (JSONB specification, fields, document type).
    3. document_markdowns:
       Stores converted Markdown outputs (.md) produced by Layer 1 & 2 conversion.
    4. extraction_runs:
       Stores structured field extraction results produced by Layer 3.

Owner: engineer-b@idp-pilot & engineer-a@idp-pilot
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def get_engine():
    return engine


def get_session():
    return SessionLocal()


# ==============================================================================
# Table 1: Uploaded Input Documents (PDFs)
# ==============================================================================
class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String, default="application/pdf")
    file_hash: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    file_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 2: Confirmed Target Schemas
# ==============================================================================
class SchemaRecord(Base):
    __tablename__ = "schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schema_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    document_type: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    sample_documents: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 3: Converted Markdown Outputs (.md)
# ==============================================================================
class MarkdownRecord(Base):
    __tablename__ = "document_markdowns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    md_filename: Mapped[str] = mapped_column(String, nullable=False)
    markdown_content: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    schema_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    schema_ref_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    pages_json: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 4: Structured Field Extraction Runs (Layer 3)
# ==============================================================================
class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schema_name: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    page_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    llm_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 5: Pipeline Job Reports (Full JSON PDF Binaries)
# ==============================================================================
class JobPdfRecord(Base):
    __tablename__ = "job_pdfs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    pdf_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    """Create all 5 tables in PostgreSQL if they do not exist."""
    Base.metadata.create_all(engine)


# ==============================================================================
# CRUD Helper Functions with Safe Error Handling
# ==============================================================================

def save_document(filename: str, file_bytes: bytes, content_type: str = "application/pdf") -> Optional[int]:
    """
    Save or update an uploaded document (PDF) in PostgreSQL.
    """
    try:
        init_db()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        session = SessionLocal()
        try:
            doc = session.query(DocumentRecord).filter_by(filename=filename).first()
            if doc:
                doc.file_size = len(file_bytes)
                doc.file_data = file_bytes
                doc.file_hash = file_hash
                doc.content_type = content_type
                doc.created_at = datetime.now(timezone.utc)
            else:
                doc = DocumentRecord(
                    filename=filename,
                    file_size=len(file_bytes),
                    content_type=content_type,
                    file_hash=file_hash,
                    file_data=file_bytes,
                )
                session.add(doc)
            session.commit()
            session.refresh(doc)
            logger.info("storage.document_saved", file_name=filename, doc_id=doc.id, size=len(file_bytes))
            return doc.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_document_failed", file_name=filename, error=str(exc))
        return None


def save_schema_record(
    schema_id: str,
    document_type: str,
    schema_json: dict,
    session_id: Optional[str] = None,
    sample_documents: Optional[list] = None,
    confirmed_at: Optional[datetime] = None,
) -> Optional[int]:
    """
    Save or update a target schema in PostgreSQL.
    """
    try:
        init_db()
        fields = (schema_json.get("schema") or {}).get("fields", []) or schema_json.get("fields", [])
        field_count = len(fields)
        session = SessionLocal()
        try:
            rec = session.query(SchemaRecord).filter_by(schema_id=schema_id).first()
            if rec:
                rec.document_type = document_type
                rec.field_count = field_count
                rec.schema_json = schema_json
                rec.sample_documents = sample_documents or []
                rec.session_id = session_id
                if confirmed_at:
                    rec.confirmed_at = confirmed_at
            else:
                rec = SchemaRecord(
                    schema_id=schema_id,
                    document_type=document_type,
                    field_count=field_count,
                    schema_json=schema_json,
                    sample_documents=sample_documents or [],
                    session_id=session_id,
                    confirmed_at=confirmed_at or datetime.now(timezone.utc),
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.schema_saved", schema_id=schema_id, id=rec.id, field_count=field_count)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_schema_failed", schema_id=schema_id, error=str(exc))
        return None


def save_markdown_record(
    doc_id: str,
    md_filename: str,
    markdown_content: str,
    schema_id: Optional[str] = None,
    schema_ref_json: Optional[dict] = None,
    page_count: int = 1,
    pages_json: Optional[list] = None,
) -> Optional[int]:
    """
    Save or update converted Markdown (.md) in PostgreSQL.
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(MarkdownRecord).filter_by(doc_id=doc_id).first()
            if rec:
                rec.md_filename = md_filename
                rec.markdown_content = markdown_content
                rec.page_count = page_count
                rec.schema_id = schema_id
                rec.schema_ref_json = schema_ref_json
                rec.pages_json = pages_json
                rec.created_at = datetime.now(timezone.utc)
            else:
                rec = MarkdownRecord(
                    doc_id=doc_id,
                    md_filename=md_filename,
                    markdown_content=markdown_content,
                    page_count=page_count,
                    schema_id=schema_id,
                    schema_ref_json=schema_ref_json,
                    pages_json=pages_json,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.markdown_saved", doc_id=doc_id, md_filename=md_filename, id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_markdown_failed", doc_id=doc_id, error=str(exc))
        return None


def save_extraction_run(
    doc_id: str,
    page_count: int | None,
    schema_name: str,
    result_json: dict,
    llm_provider: str | None,
    model_name: str | None,
    processing_time_seconds: float | None,
    page_outputs: list | None = None,
) -> Optional[int]:
    """
    Save one extraction run in PostgreSQL.
    """
    try:
        init_db()
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
            logger.info("storage.extraction_saved", doc_id=doc_id, run_id=run.id)
            return run.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_extraction_failed", doc_id=doc_id, error=str(exc))
        return None


def save_job_pdf(job_id: str, pdf_bytes: bytes, filename: Optional[str] = None) -> Optional[int]:
    """
    Save or update a pipeline job PDF report in PostgreSQL.
    """
    if not filename:
        filename = f"{job_id}_report.pdf"
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(JobPdfRecord).filter_by(job_id=job_id).first()
            if rec:
                rec.filename = filename
                rec.pdf_data = pdf_bytes
                rec.created_at = datetime.now(timezone.utc)
            else:
                rec = JobPdfRecord(
                    job_id=job_id,
                    filename=filename,
                    pdf_data=pdf_bytes,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.job_pdf_saved", job_id=job_id, size=len(pdf_bytes), id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_job_pdf_failed", job_id=job_id, error=str(exc))
        return None


def get_job_pdf(job_id: str) -> Optional[tuple[bytes, str]]:
    """
    Retrieve stored job PDF binary and filename from PostgreSQL.
    Returns (pdf_bytes, filename) or None if not found.
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(JobPdfRecord).filter_by(job_id=job_id).first()
            if rec:
                return (bytes(rec.pdf_data), rec.filename)
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_job_pdf_failed", job_id=job_id, error=str(exc))
        return None
