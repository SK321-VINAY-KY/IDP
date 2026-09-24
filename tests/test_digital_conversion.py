"""
File: test_digital_conversion.py
Purpose: Regression tests for DocumentConverter singleton caching in Layer 2 digital path.
         Verifies that repeated calls to convert_digital_page reuse the same
         DocumentConverter instance rather than re-instantiating on each invocation.
"""
from unittest.mock import MagicMock, patch
import pytest

import src.ai.layer2_conversion.digital as digital


@pytest.fixture(autouse=True)
def reset_digital_singleton():
    """Ensure clean singleton state before and after each test."""
    digital._docling_converter = None
    digital.DocumentConverter = None
    yield
    digital._docling_converter = None
    digital.DocumentConverter = None


def test_convert_digital_page_reuses_document_converter_instance():
    """
    Verify that convert_digital_page() reuses the DocumentConverter singleton
    across multiple page conversion calls, instantiating DocumentConverter() only once.
    """
    mock_instance = MagicMock()
    mock_result_p1 = MagicMock()
    mock_result_p1.document.export_to_markdown.return_value = "Page 1: Digital text extraction content exceeding threshold."
    mock_result_p2 = MagicMock()
    mock_result_p2.document.export_to_markdown.return_value = "Page 2: Digital text extraction content exceeding threshold."

    def convert_side_effect(pdf_path, page_range):
        if page_range == (1, 1):
            return mock_result_p1
        return mock_result_p2

    mock_instance.convert.side_effect = convert_side_effect

    with patch("src.ai.layer2_conversion.digital.DocumentConverter", return_value=mock_instance) as mock_cls:
        # First call: should instantiate DocumentConverter once and convert page 1
        md1, conf1 = digital.convert_digital_page("test_doc.pdf", 1)

        # Second call: should reuse existing DocumentConverter without re-instantiating
        md2, conf2 = digital.convert_digital_page("test_doc.pdf", 2)

        # Third call: another page
        md3, conf3 = digital.convert_digital_page("test_doc.pdf", 3)

        # Verify DocumentConverter() was instantiated exactly ONCE
        assert mock_cls.call_count == 1, f"Expected 1 instantiation, got {mock_cls.call_count}"

        # Verify all calls converted successfully using the same converter instance
        assert mock_instance.convert.call_count == 3
        mock_instance.convert.assert_any_call("test_doc.pdf", page_range=(1, 1))
        mock_instance.convert.assert_any_call("test_doc.pdf", page_range=(2, 2))
        mock_instance.convert.assert_any_call("test_doc.pdf", page_range=(3, 3))

        # Verify outputs and confidence scores
        assert "Page 1" in md1
        assert conf1 == 0.97
        assert "Page 2" in md2
        assert conf2 == 0.97
        assert conf3 == 0.97


def test_convert_digital_page_reuses_converter_with_docling_module_patch():
    """
    Verify singleton reuse when patching docling.document_converter.DocumentConverter.
    """
    mock_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "Extracted digital document content."
    mock_instance.convert.return_value = mock_result

    with patch("docling.document_converter.DocumentConverter", return_value=mock_instance) as mock_cls:
        md1, conf1 = digital.convert_digital_page("sample.pdf", 1)
        md2, conf2 = digital.convert_digital_page("sample.pdf", 2)

        assert mock_cls.call_count == 1
        assert mock_instance.convert.call_count == 2
        assert md1 == "Extracted digital document content."
        assert md2 == "Extracted digital document content."
        assert conf1 == 0.97
        assert conf2 == 0.97


def test_convert_digital_page_fallback_when_docling_unavailable():
    """
    Verify fallback to PyMuPDF when docling cannot be imported.
    """
    with patch.object(digital, "_get_docling_converter", side_effect=ImportError("No module named 'docling'")):
        with patch("pymupdf.open") as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Fallback text extracted via PyMuPDF successfully."
            mock_doc.__getitem__.return_value = mock_page
            mock_open.return_value = mock_doc

            md, conf = digital.convert_digital_page("fallback.pdf", 1)

            assert md == "Fallback text extracted via PyMuPDF successfully."
            assert conf == 0.95
            mock_open.assert_called_once_with("fallback.pdf")
            mock_doc.close.assert_called_once()
