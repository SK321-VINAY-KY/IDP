from app.core.schema_state import SchemaState
from app.core.validator import validate_schema


def test_valid_schema_has_no_errors():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("invoice_number", type="string", required=True)
    s.add_field("total_amount", type="number", required=True)
    assert validate_schema(s) == []


def test_missing_document_type_is_an_error():
    s = SchemaState()
    s.add_field("x", type="string", required=True)
    errors = validate_schema(s)
    assert any("document_type" in e for e in errors)


def test_no_fields_is_an_error():
    s = SchemaState()
    s.set_document_type("invoice")
    errors = validate_schema(s)
    assert any("no fields" in e for e in errors)


def test_unsupported_type_is_an_error():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("x", type="varchar", required=True)
    errors = validate_schema(s)
    assert any("unsupported type" in e for e in errors)


def test_array_without_item_type_is_an_error():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("items", type="array", required=True)
    errors = validate_schema(s)
    assert any("item_type" in e for e in errors)


def test_missing_required_flag_is_an_error():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("x", type="string")
    errors = validate_schema(s)
    assert any("required" in e for e in errors)
