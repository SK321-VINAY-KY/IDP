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
import json
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    JSON,
    create_engine,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reserved sentinel owner values for system isolation
SENTINEL_LEGACY = "_legacy"
SENTINEL_ADMIN = "_admin"


class Base(DeclarativeBase):
    pass


def _normalize_pg_url(db_url: str) -> str:
    """
    Pin the PostgreSQL dialect to psycopg2, which is the driver installed in
    the image. SQLAlchemy 2.1 changed the default driver for a bare
    'postgresql://' URL to psycopg (v3); without an explicit '+psycopg2' the
    engine would try to import psycopg (v3), fail, and silently fall back to
    SQLite. Normalizing here keeps existing DATABASE_URL env values working.
    """
    if db_url.startswith("postgresql+"):
        return db_url  # driver already specified explicitly
    if db_url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + db_url[len("postgresql://"):]
    if db_url.startswith("postgres://"):
        return "postgresql+psycopg2://" + db_url[len("postgres://"):]
    return db_url


def _create_db_engine():
    db_url = getattr(settings, "database_url", "sqlite:///./idp_storage.db")
    try:
        if db_url.startswith("postgres"):
            db_url = _normalize_pg_url(db_url)
            eng = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            # Ping test connection
            with eng.connect() as conn:
                pass
            return eng
        return create_engine(db_url)
    except Exception as exc:
        logger.warning("storage.engine_init_fallback", error=str(exc))
        return create_engine("sqlite:///./idp_storage.db")


engine = _create_db_engine()
SessionLocal = sessionmaker(bind=engine)


def get_engine():
    return engine


def get_session():
    return SessionLocal()


# ==============================================================================
# Table 1: Uploaded Input Documents (PDFs)
# Shadow copy for input documents; non-authoritative when S3 is configured.
# Keyed by (owner, filename).
# ==============================================================================
class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("owner", "filename", name="uq_documents_owner_filename"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False, index=True)
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
    owner: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    schema_name: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    sample_documents: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 3: Converted Markdown Outputs (.md)
# Keyed by (owner, doc_id).
# ==============================================================================
class MarkdownRecord(Base):
    __tablename__ = "document_markdowns"
    __table_args__ = (
        UniqueConstraint("owner", "doc_id", name="uq_document_markdowns_owner_doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    md_filename: Mapped[str] = mapped_column(String, nullable=False)
    markdown_content: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    schema_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    schema_ref_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pages_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 4: Structured Field Extraction Runs (Layer 3)
# Indexed by (owner, doc_id).
# ==============================================================================
class ExtractionRun(Base):
    __tablename__ = "extraction_runs"
    __table_args__ = (
        Index("ix_extraction_runs_owner_doc_id", "owner", "doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schema_name: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    page_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
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


# ==============================================================================
# Table 6: Document Knowledge Graph Memory (Layer 3)
# ==============================================================================
class DocumentGraphRecord(Base):
    __tablename__ = "document_graphs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    job_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    owner: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    schema_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    strategy: Mapped[str] = mapped_column(String, default="graph_memory")
    graph_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 7: Durable Page Delta Checkpoints (Layer 3)
# ==============================================================================
class PageCheckpointRecord(Base):
    __tablename__ = "page_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_key: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String, default="_admin", nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pipeline_version: Mapped[str] = mapped_column(String, default="v1")
    extraction_version: Mapped[str] = mapped_column(String, default="v1")
    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    schema_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    page_delta_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    delta_checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    error_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table 8: Users (Layer 3 Auth Persistence)
# ==============================================================================
class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


def init_db():
    """Create all 8 tables in PostgreSQL / SQLite if they do not exist."""
    global engine, SessionLocal
    try:
        Base.metadata.create_all(engine)
    except Exception as exc:
        logger.warning("storage.init_db_fallback", error=str(exc))
        engine = create_engine("sqlite:///./idp_storage.db")
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)


# ==============================================================================
# CRUD Helper Functions with Safe Error Handling
# ==============================================================================

def save_document(filename: str, file_bytes: bytes, owner: str, content_type: str = "application/pdf") -> Optional[int]:
    """
    Save or update an uploaded document (PDF) shadow copy in PostgreSQL / SQLite.
    Non-authoritative when S3 is configured. Keyed by (owner, filename).
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        session = SessionLocal()
        try:
            doc = session.query(DocumentRecord).filter_by(owner=owner, filename=filename).first()
            if doc:
                doc.file_size = len(file_bytes)
                doc.file_data = file_bytes
                doc.file_hash = file_hash
                doc.content_type = content_type
                doc.created_at = datetime.now(timezone.utc)
            else:
                doc = DocumentRecord(
                    owner=owner,
                    filename=filename,
                    file_size=len(file_bytes),
                    content_type=content_type,
                    file_hash=file_hash,
                    file_data=file_bytes,
                )
                session.add(doc)
            session.commit()
            session.refresh(doc)
            logger.info("storage.document_saved", owner=owner, file_name=filename, doc_id=doc.id, size=len(file_bytes))
            return doc.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_document_failed", owner=owner, file_name=filename, error=str(exc))
        return None


def save_schema_record(
    schema_id: str,
    document_type: str,
    schema_json: dict,
    session_id: Optional[str] = None,
    sample_documents: Optional[list] = None,
    confirmed_at: Optional[datetime] = None,
    owner: Optional[str] = None,
    schema_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Save or insert a target schema in PostgreSQL.

    Versioning: schemas are keyed by (owner, schema_name). Each new confirmation
    for the same owner+schema_name gets an auto-incremented version (v1, v2, ...).
    Returns a dict {id, version, schema_id} on success, or None on failure.
    """
    try:
        init_db()
        fields = (schema_json.get("schema") or {}).get("fields", []) or schema_json.get("fields", [])
        field_count = len(fields)
        # Fall back to document_type as the logical name if none supplied.
        eff_name = (schema_name or document_type or "schema").strip()
        session = SessionLocal()
        try:
            rec = session.query(SchemaRecord).filter_by(schema_id=schema_id).first()
            if rec:
                # Update-in-place (same schema_id): keep its version.
                rec.document_type = document_type
                rec.owner = owner if owner is not None else rec.owner
                rec.schema_name = eff_name
                rec.field_count = field_count
                rec.schema_json = schema_json
                rec.sample_documents = sample_documents or []
                rec.session_id = session_id
                if confirmed_at:
                    rec.confirmed_at = confirmed_at
                version = rec.version or 1
            else:
                # New schema_id: compute next version for this (owner, schema_name).
                max_v = (
                    session.query(SchemaRecord.version)
                    .filter(SchemaRecord.owner == owner, SchemaRecord.schema_name == eff_name)
                    .order_by(SchemaRecord.version.desc())
                    .first()
                )
                version = ((max_v[0] if max_v and max_v[0] else 0) + 1)
                # Reflect owner/name/version inside the stored json too (for readers).
                if isinstance(schema_json, dict):
                    schema_json = dict(schema_json)
                    schema_json.setdefault("owner", owner)
                    schema_json["schema_name"] = eff_name
                    schema_json["version"] = version
                rec = SchemaRecord(
                    schema_id=schema_id,
                    document_type=document_type,
                    owner=owner,
                    schema_name=eff_name,
                    version=version,
                    field_count=field_count,
                    schema_json=schema_json,
                    sample_documents=sample_documents or [],
                    session_id=session_id,
                    confirmed_at=confirmed_at or datetime.now(timezone.utc),
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.schema_saved", schema_id=schema_id, id=rec.id, owner=owner, schema_name=eff_name, version=rec.version, field_count=field_count)
            return {"id": rec.id, "version": rec.version, "schema_id": rec.schema_id}
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_schema_failed", schema_id=schema_id, error=str(exc))
        return None


def save_markdown_record(
    doc_id: str,
    md_filename: str,
    markdown_content: str,
    owner: str,
    schema_id: Optional[str] = None,
    schema_ref_json: Optional[dict] = None,
    page_count: int = 1,
    pages_json: Optional[list] = None,
) -> Optional[int]:
    """
    Save or update converted Markdown (.md) in PostgreSQL / SQLite.
    Keyed by (owner, doc_id).
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(MarkdownRecord).filter_by(owner=owner, doc_id=doc_id).first()
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
                    owner=owner,
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
            logger.info("storage.markdown_saved", owner=owner, doc_id=doc_id, md_filename=md_filename, id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_markdown_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def get_markdown_record(doc_id: str, owner: str) -> Optional[dict]:
    """
    Retrieve stored markdown record by doc_id or md_filename, strictly scoped to owner.
    Candidate normalization is executed strictly WITHIN the owner scope.
    """
    if not doc_id or not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            from pathlib import Path as _Path
            stem = _Path(doc_id).stem
            candidates = list(dict.fromkeys([doc_id, stem, f"{stem}.pdf", f"{stem}.md"]))
            rec = (
                session.query(MarkdownRecord)
                .filter(
                    MarkdownRecord.owner == owner,
                    (MarkdownRecord.doc_id.in_(candidates) | MarkdownRecord.md_filename.in_(candidates)),
                )
                .order_by(MarkdownRecord.id.desc())
                .first()
            )
            if rec:
                return {
                    "id": rec.id,
                    "owner": rec.owner,
                    "doc_id": rec.doc_id,
                    "md_filename": rec.md_filename,
                    "markdown_content": rec.markdown_content,
                    "page_count": rec.page_count,
                    "schema_id": rec.schema_id,
                    "schema_ref_json": rec.schema_ref_json,
                    "pages_json": rec.pages_json,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                }
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_markdown_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def save_extraction_run(
    doc_id: str,
    page_count: int | None,
    schema_name: str,
    result_json: dict,
    owner: str,
    llm_provider: str | None = None,
    model_name: str | None = None,
    processing_time_seconds: float | None = None,
    page_outputs: list | None = None,
) -> Optional[int]:
    """
    Save one extraction run in PostgreSQL / SQLite, strictly associated with owner.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
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
                owner=owner,
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
            logger.info("storage.extraction_saved", owner=owner, doc_id=doc_id, run_id=run.id)
            return run.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_extraction_failed", owner=owner, doc_id=doc_id, error=str(exc))
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
                return (rec.pdf_data, rec.filename)
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_job_pdf_failed", job_id=job_id, error=str(exc))
        return None


# ==============================================================================
# Document Graph Persistence (Layer 3)
# ==============================================================================

def save_document_graph(
    doc_id: str,
    graph_dict: dict,
    job_id: Optional[str] = None,
    owner: Optional[str] = None,
    schema_id: Optional[str] = None,
    strategy: str = "graph_memory",
) -> Optional[int]:
    """
    Save or update a document's GraphMemory representation in PostgreSQL / SQLite.
    Associated with doc_id, job_id, and owner for strict multi-tenant isolation.
    """
    try:
        init_db()
        nodes = graph_dict.get("nodes", [])
        edges = graph_dict.get("edges", [])
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord).filter_by(doc_id=doc_id)
            if job_id:
                query = query.filter_by(job_id=job_id)
            rec = query.first()

            if rec:
                rec.graph_json = graph_dict
                rec.node_count = len(nodes)
                rec.edge_count = len(edges)
                rec.strategy = strategy
                if owner:
                    rec.owner = owner
                if schema_id:
                    rec.schema_id = schema_id
                rec.updated_at = datetime.now(timezone.utc)
            else:
                rec = DocumentGraphRecord(
                    doc_id=doc_id,
                    job_id=job_id,
                    owner=owner,
                    schema_id=schema_id,
                    strategy=strategy,
                    graph_json=graph_dict,
                    node_count=len(nodes),
                    edge_count=len(edges),
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info("storage.document_graph_saved", doc_id=doc_id, job_id=job_id, nodes=len(nodes), edges=len(edges), id=rec.id)
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.save_document_graph_failed", doc_id=doc_id, error=str(exc))
        return None


def get_document_graph(
    doc_id: Optional[str] = None,
    job_id: Optional[str] = None,
    owner: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve stored document graph dictionary.
    Enforces user isolation if owner is provided (non-admin).
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord)
            if doc_id:
                query = query.filter_by(doc_id=doc_id)
                if job_id:
                    query = query.filter_by(job_id=job_id)
            elif job_id:
                query = query.filter_by(job_id=job_id)
            else:
                return None

            if owner and owner != "admin":
                query = query.filter((DocumentGraphRecord.owner == owner) | (DocumentGraphRecord.owner == None))

            rec = query.order_by(DocumentGraphRecord.created_at.desc()).first()
            if rec and rec.graph_json:
                return rec.graph_json
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_document_graph_failed", doc_id=doc_id, job_id=job_id, error=str(exc))
        return None


def list_document_graphs(owner: Optional[str] = None) -> list:
    """
    List all available document graphs with summary metadata.
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(DocumentGraphRecord)
            if owner and owner != "admin":
                query = query.filter((DocumentGraphRecord.owner == owner) | (DocumentGraphRecord.owner == None))
            recs = query.order_by(DocumentGraphRecord.created_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "doc_id": r.doc_id,
                    "job_id": r.job_id,
                    "owner": r.owner,
                    "schema_id": r.schema_id,
                    "strategy": r.strategy,
                    "node_count": r.node_count,
                    "edge_count": r.edge_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recs
            ]
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_document_graphs_failed", error=str(exc))
        return []


# ==============================================================================
# Page Checkpoint Helpers (Layer 3 Durable Recovery)
# ==============================================================================

def compute_page_delta_checksum(delta_dict: Optional[dict]) -> str:
    """Compute deterministic SHA256 checksum of a PageDelta dictionary."""
    if not delta_dict:
        return ""
    serialized = json.dumps(delta_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_checkpoint_key(
    doc_id: str,
    page_number: int,
    owner: str,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
) -> str:
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    return f"{owner.strip()}:{doc_id}:p{page_number}:{pipeline_version}:{extraction_version}"


def save_page_checkpoint(
    doc_id: str,
    page_number: int,
    status: str,
    owner: str,
    page_delta_json: Optional[dict] = None,
    delta_checksum: Optional[str] = None,
    error_info: Optional[str] = None,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
    model_version: Optional[str] = None,
    prompt_version: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> Optional[int]:
    """
    Save or update a durable page checkpoint in PostgreSQL / SQLite.
    Protects against duplicate page processing and enables crash recovery.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner is required and cannot be empty")
    owner = owner.strip()
    try:
        init_db()
        key = build_checkpoint_key(doc_id, page_number, owner=owner, pipeline_version=pipeline_version, extraction_version=extraction_version)
        if page_delta_json and not delta_checksum:
            delta_checksum = compute_page_delta_checksum(page_delta_json)

        session = SessionLocal()
        try:
            rec = session.query(PageCheckpointRecord).filter_by(checkpoint_key=key).first()
            if rec:
                rec.status = status
                rec.owner = owner
                if page_delta_json is not None:
                    rec.page_delta_json = page_delta_json
                if delta_checksum:
                    rec.delta_checksum = delta_checksum
                if error_info is not None:
                    rec.error_info = error_info
                rec.attempt_count += 1
                rec.updated_at = datetime.now(timezone.utc)
            else:
                rec = PageCheckpointRecord(
                    checkpoint_key=key,
                    owner=owner,
                    doc_id=doc_id,
                    page_number=page_number,
                    pipeline_version=pipeline_version,
                    extraction_version=extraction_version,
                    model_version=model_version,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    status=status,
                    page_delta_json=page_delta_json,
                    delta_checksum=delta_checksum,
                    attempt_count=1,
                    error_info=error_info,
                )
                session.add(rec)
            session.commit()
            session.refresh(rec)
            logger.info(
                "page.checkpoint.created",
                checkpoint_key=key,
                owner=owner,
                doc_id=doc_id,
                page_number=page_number,
                status=status,
                attempt=rec.attempt_count,
            )
            return rec.id
        finally:
            session.close()
    except Exception as exc:
        logger.error(
            "page.checkpoint.save_failed",
            owner=owner,
            doc_id=doc_id,
            page_number=page_number,
            error=str(exc),
        )
        return None


def get_page_checkpoint(
    doc_id: str,
    page_number: int,
    owner: str,
    pipeline_version: str = "v1",
    extraction_version: str = "v1",
) -> Optional[dict]:
    """Retrieve durable page checkpoint dictionary."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        init_db()
        key = build_checkpoint_key(doc_id, page_number, owner=owner, pipeline_version=pipeline_version, extraction_version=extraction_version)
        session = SessionLocal()
        try:
            rec = session.query(PageCheckpointRecord).filter_by(checkpoint_key=key).first()
            if rec:
                return {
                    "id": rec.id,
                    "checkpoint_key": rec.checkpoint_key,
                    "owner": rec.owner,
                    "doc_id": rec.doc_id,
                    "page_number": rec.page_number,
                    "pipeline_version": rec.pipeline_version,
                    "extraction_version": rec.extraction_version,
                    "status": rec.status,
                    "page_delta_json": rec.page_delta_json,
                    "delta_checksum": rec.delta_checksum,
                    "attempt_count": rec.attempt_count,
                    "error_info": rec.error_info,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                    "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
                }
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_page_checkpoint_failed", owner=owner, doc_id=doc_id, page=page_number, error=str(exc))
        return None


def list_page_checkpoints(doc_id: str, owner: str, pipeline_version: Optional[str] = None) -> List[dict]:
    """List all checkpoints for a document in ascending page order."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return []
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(PageCheckpointRecord).filter_by(doc_id=doc_id, owner=owner)
            if pipeline_version:
                query = query.filter_by(pipeline_version=pipeline_version)
            recs = query.order_by(PageCheckpointRecord.page_number.asc()).all()
            return [
                {
                    "id": r.id,
                    "checkpoint_key": r.checkpoint_key,
                    "owner": r.owner,
                    "doc_id": r.doc_id,
                    "page_number": r.page_number,
                    "pipeline_version": r.pipeline_version,
                    "extraction_version": r.extraction_version,
                    "status": r.status,
                    "page_delta_json": r.page_delta_json,
                    "delta_checksum": r.delta_checksum,
                    "attempt_count": r.attempt_count,
                    "error_info": r.error_info,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in recs
            ]
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_page_checkpoints_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return []


def delete_page_checkpoints(doc_id: str, owner: str) -> int:
    """Delete all checkpoints for a given document."""
    if not owner or not isinstance(owner, str) or not owner.strip():
        return 0
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            deleted = session.query(PageCheckpointRecord).filter_by(doc_id=doc_id, owner=owner).delete()
            session.commit()
            return deleted
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.delete_page_checkpoints_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return 0


# ==============================================================================
# Schema Read Helpers (DB-backed reads for the web app)
# ==============================================================================

def _schema_record_to_dict(rec: "SchemaRecord") -> dict:
    """
    Normalize a SchemaRecord row into the same shape the web app previously
    read from schema_registry/<id>.json. The full original record is stored in
    schema_json, so return that as the authoritative payload, augmented with
    top-level summary fields the endpoints rely on.
    """
    payload = dict(rec.schema_json) if isinstance(rec.schema_json, dict) else {}
    # Ensure the summary fields are always present and consistent with the row.
    payload.setdefault("schema_id", rec.schema_id)
    payload.setdefault("document_type", rec.document_type)
    if rec.confirmed_at and not payload.get("confirmed_at"):
        payload["confirmed_at"] = rec.confirmed_at.isoformat()
    return payload


def get_schema_record(schema_id: str, owner: Optional[str] = None) -> Optional[dict]:
    """
    Retrieve a single confirmed schema record from PostgreSQL by schema_id.
    Returns the full stored schema_json dict (same shape as the old on-disk
    schema_registry/<id>.json), or None if not found.

    If ``owner`` is provided, the returned record must belong to that owner
    (i.e. rec.owner == owner).  Records with a NULL owner are shared/admin
    schemas and are visible to all callers regardless of owner.
    Pass owner=None (default) to skip owner-scoping (admin-only code paths).
    """
    if not schema_id:
        return None
    try:
        init_db()
        session = SessionLocal()
        try:
            rec = session.query(SchemaRecord).filter_by(schema_id=schema_id).first()
            if not rec:
                # Fall back to disk schema_registry/
                from pathlib import Path
                reg_file = Path("schema_registry") / f"{schema_id}.json"
                if reg_file.is_file():
                    try:
                        data = json.loads(reg_file.read_text(encoding="utf-8"))
                        # Owner check on disk fallback
                        if owner is not None:
                            rec_owner = data.get("owner")
                            if rec_owner is not None and rec_owner != owner:
                                return None
                        return data
                    except Exception:
                        pass
                return None
            # Owner check on DB record
            if owner is not None and rec.owner is not None and rec.owner != owner:
                return None
            return _schema_record_to_dict(rec)
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_schema_record_failed", schema_id=schema_id, error=str(exc))
        return None



def schema_exists(schema_id: str) -> bool:
    """Return True if a schema with this schema_id exists in the DB."""
    if not schema_id:
        return False
    try:
        init_db()
        session = SessionLocal()
        try:
            return session.query(SchemaRecord).filter_by(schema_id=schema_id).first() is not None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.schema_exists_failed", schema_id=schema_id, error=str(exc))
        return False


def list_schema_records(owner: Optional[str] = None) -> List[dict]:
    """
    List confirmed schemas from PostgreSQL as summary dicts, newest first.

    Ordering: by confirmed_at desc (falling back to created_at desc).
    Owner filtering: if `owner` is provided, returns schemas whose stored
    record has that owner OR has no owner set (legacy/ownerless records remain
    visible, mirroring the previous disk behavior). Pass owner=None for admin
    (returns everything).
    """
    try:
        init_db()
        session = SessionLocal()
        try:
            q = session.query(SchemaRecord)
            if owner is not None:
                # Strict per-user isolation: only schemas owned by this user.
                # Falls back to the owner embedded in schema_json for legacy rows.
                q = q.filter(SchemaRecord.owner == owner)
            recs = q.order_by(
                SchemaRecord.confirmed_at.desc().nullslast(), SchemaRecord.created_at.desc()
            ).all()
            results: List[dict] = []
            for rec in recs:
                data = rec.schema_json if isinstance(rec.schema_json, dict) else {}
                rec_owner = rec.owner or data.get("owner")
                # Legacy rows may have no owner column yet; enforce json owner too.
                if owner is not None and rec.owner is None and data.get("owner") not in (None, owner):
                    continue
                inner = data.get("schema") if isinstance(data.get("schema"), dict) else {}
                doc_type = inner.get("document_type") or data.get("document_type") or rec.document_type
                field_count = rec.field_count
                if not field_count:
                    field_count = len(inner.get("fields", []) or data.get("fields", []) or [])
                results.append({
                    "schema_id": rec.schema_id,
                    "document_type": doc_type,
                    "schema_name": rec.schema_name or data.get("schema_name") or doc_type,
                    "version": rec.version or data.get("version") or 1,
                    "confirmed_at": data.get("confirmed_at") or (rec.confirmed_at.isoformat() if rec.confirmed_at else None),
                    "field_count": field_count,
                    "sample_documents": rec.sample_documents or data.get("sample_documents", []) or [],
                    "owner": rec_owner,
                })
            return results
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_schema_records_failed", error=str(exc))
        return []


def get_latest_owned_schema_id(owner: str) -> Optional[str]:
    """
    Return the schema_id of the most recently confirmed schema owned by `owner`
    (used by the zero-parameter user pipeline trigger). Ownerless schemas are
    not considered "owned" here, matching the previous disk semantics which
    filtered strictly on owner == username.
    """
    if not owner:
        return None
    try:
        init_db()
        session = SessionLocal()
        try:
            recs = (
                session.query(SchemaRecord)
                .order_by(SchemaRecord.confirmed_at.desc().nullslast(), SchemaRecord.created_at.desc())
                .all()
            )
            for rec in recs:
                data = rec.schema_json if isinstance(rec.schema_json, dict) else {}
                if (rec.owner or data.get("owner")) == owner:
                    return rec.schema_id

            # Fall back to disk schema_registry/
            from pathlib import Path
            reg_dir = Path("schema_registry")
            if reg_dir.exists():
                candidates = []
                for f in reg_dir.glob("*.json"):
                    try:
                        sdata = json.loads(f.read_text(encoding="utf-8"))
                        if sdata.get("owner") == owner:
                            conf = sdata.get("confirmed_at") or ""
                            candidates.append((conf, sdata.get("schema_id") or f.stem))
                    except Exception:
                        pass
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]

            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_latest_owned_schema_failed", owner=owner, error=str(exc))
        return None


# ==============================================================================
# Extraction Run Read Helpers (DB-backed reads for the Query Bot JSON fallback)
# ==============================================================================

def get_latest_extracted_data(doc_id: str, owner: str) -> Optional[dict]:
    """
    Retrieve the most recent extraction result for a document from PostgreSQL / SQLite.
    Candidate matching is strictly evaluated WITHIN the specified owner scope.
    """
    if not doc_id or not owner or not isinstance(owner, str) or not owner.strip():
        return None
    owner = owner.strip()
    try:
        from pathlib import Path as _Path
        stem = _Path(doc_id).stem
        candidates = list(dict.fromkeys([doc_id, stem, f"{stem}.pdf"]))
        init_db()
        session = SessionLocal()
        try:
            rec = (
                session.query(ExtractionRun)
                .filter(
                    ExtractionRun.owner == owner,
                    ExtractionRun.doc_id.in_(candidates),
                )
                .order_by(ExtractionRun.created_at.desc())
                .first()
            )
            if not rec or not rec.result_json:
                return None
            result = rec.result_json
            if isinstance(result, dict):
                # Prefer the nested extracted_data if present, else return as-is.
                return result.get("extracted_data") or result
            return None
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.get_latest_extracted_data_failed", owner=owner, doc_id=doc_id, error=str(exc))
        return None


def list_markdown_records(owner: str) -> List[dict]:
    """
    List converted-markdown output records from PostgreSQL / SQLite as summary dicts,
    newest first. Strictly filtered by owner.
    Admins can pass '_all' or '*' to inspect across all owners.
    """
    if not owner or not isinstance(owner, str) or not owner.strip():
        return []
    owner = owner.strip()
    try:
        init_db()
        session = SessionLocal()
        try:
            query = session.query(MarkdownRecord)
            if owner not in ("*", "_all"):
                query = query.filter(MarkdownRecord.owner == owner)
            recs = query.order_by(MarkdownRecord.created_at.desc()).all()
            out: List[dict] = []
            seen = set()
            for r in recs:
                # De-duplicate by (owner, md_filename)
                key = (r.owner, r.md_filename or f"{r.doc_id}.md")
                if key in seen:
                    continue
                seen.add(key)
                ref = r.schema_ref_json if isinstance(r.schema_ref_json, dict) else {}
                out.append({
                    "name": r.md_filename or f"{r.doc_id}.md",
                    "doc_id": r.doc_id,
                    "owner": r.owner,
                    "size": len(r.markdown_content or ""),
                    "modified": r.created_at.isoformat() if r.created_at else None,
                    "page_count": r.page_count,
                    "schema_id": r.schema_id or ref.get("schema_id"),
                    "source_pdf": ref.get("source_pdf") or (
                        f"{r.doc_id}" if str(r.doc_id).lower().endswith(".pdf") else f"{r.doc_id}.pdf"
                    ),
                    "schema_ref": ref or None,
                })
            return out
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.list_markdown_records_failed", owner=owner, error=str(exc))
        return []


def _owned_doc_ids(owner: str) -> set:
    """
    Return the set of doc_id variants owned by `owner`, derived from
    DocumentGraphRecord ownership (the authoritative per-user marker).
    """
    owned = set()
    if not owner:
        return owned
    try:
        init_db()
        session = SessionLocal()
        try:
            from pathlib import Path as _Path
            recs = session.query(DocumentGraphRecord).filter(DocumentGraphRecord.owner == owner).all()
            for r in recs:
                d = r.doc_id or ""
                stem = _Path(d).stem
                owned.update({d, stem, f"{stem}.pdf"})
        finally:
            session.close()
    except Exception as exc:
        logger.error("storage.owned_doc_ids_failed", owner=owner, error=str(exc))
    return owned


def list_markdown_records_for_owner(owner: str) -> List[dict]:
    """
    List processed-output (markdown) summaries that belong to `owner`.
    Directly delegates to list_markdown_records(owner=owner) enforcing DB isolation.
    """
    return list_markdown_records(owner=owner)
