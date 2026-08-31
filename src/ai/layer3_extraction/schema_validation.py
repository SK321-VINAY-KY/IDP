"""
File: schema_validation.py
Purpose: Retry/repair loop for extraction validation failures.
Owner: engineer-b@idp-pilot
Created: 2026-08-20
"""
from typing import Callable, Optional
from pydantic import BaseModel, ValidationError

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_with_retry(extract_fn: Callable[[], BaseModel], schema: type[BaseModel]) -> BaseModel:
    last_error: Optional[Exception] = None
    for attempt in range(1, settings.max_extraction_retries + 2):
        try:
            result = extract_fn()
            schema.model_validate(result.model_dump())
            if attempt > 1:
                logger.info("extraction.recovered_after_retry", attempt=attempt)
            return result
        except (ValidationError, ValueError) as e:
            last_error = e
            logger.warning("extraction.validation_failed", attempt=attempt, error=str(e))

    logger.error("extraction.failed_after_retries", attempts=settings.max_extraction_retries + 1)
    raise last_error